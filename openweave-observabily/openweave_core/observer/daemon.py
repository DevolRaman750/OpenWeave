"""The continuous observer: poll Langfuse -> analyze -> persist, forever.

This is what turns OpenWeave from a per-trace CLI into a production service
that watches real agents and multi-agent systems. Each tick it lists traces
emitted since the cursor, skips ones already processed or still streaming
spans, and runs the full anomaly+classification(+eval) pipeline on the rest
with bounded concurrency. Results go to the configured sinks and the durable
cursor advances so a restart resumes exactly where it left off.

Designed to be resilient: one bad trace is logged and skipped, a Langfuse
outage backs off and retries, and SIGINT/SIGTERM drain in-flight work cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from openweave_core.observer.config import ObserverConfig
from openweave_core.observer.sink import Sink, build_sink
from openweave_core.observer.state import ObserverState

log = logging.getLogger("openweave.observer")


@dataclass
class ObserverStats:
    ticks: int = 0
    discovered: int = 0
    analyzed: int = 0
    flagged: int = 0
    errors: int = 0
    last_error: Optional[str] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "discovered": self.discovered,
            "analyzed": self.analyzed,
            "flagged": self.flagged,
            "errors": self.errors,
            "last_error": self.last_error,
            "started_at": self.started_at,
        }


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class TraceObserver:
    def __init__(
        self,
        config: Optional[ObserverConfig] = None,
        sink: Optional[Sink] = None,
    ) -> None:
        self.config = config or ObserverConfig.from_env()
        self.state = ObserverState(self.config.state_path)
        self.sink = sink or build_sink(self.config)
        self.stats = ObserverStats()
        self._stop = asyncio.Event()
        self._sem = asyncio.Semaphore(max(1, self.config.concurrency))

    # -- lifecycle ---------------------------------------------------------
    def request_stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        cfg = self.config
        log.info(
            "%s starting: interval=%.1fs concurrency=%d run_eval=%s settle=%ds "
            "lookback=%ds state=%s",
            cfg.service_name,
            cfg.poll_interval_seconds,
            cfg.concurrency,
            cfg.run_eval,
            cfg.settle_seconds,
            cfg.lookback_seconds,
            cfg.state_path,
        )
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001 — a tick failure must not kill the loop
                self.stats.errors += 1
                self.stats.last_error = str(exc)
                log.exception("observer tick failed: %s", exc)
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=cfg.poll_interval_seconds
                )
            except asyncio.TimeoutError:
                pass
        log.info("%s stopped: %s", cfg.service_name, self.stats.as_dict())

    # -- one polling cycle -------------------------------------------------
    async def tick(self) -> int:
        cfg = self.config
        self.stats.ticks += 1

        now = datetime.now(timezone.utc)
        to_ts = now - timedelta(seconds=cfg.settle_seconds)
        if self.state.last_timestamp:
            from_ts = self.state.last_timestamp
        else:
            from_ts = _iso(now - timedelta(seconds=cfg.lookback_seconds))

        summaries = await self._list_new_traces(from_ts, _iso(to_ts))
        if not summaries:
            return 0

        self.stats.discovered += len(summaries)
        log.info("tick %d: %d new trace(s) to analyze", self.stats.ticks, len(summaries))

        results = await asyncio.gather(
            *(self._process_one(s) for s in summaries), return_exceptions=True
        )
        processed = sum(1 for r in results if r is True)

        # Persist cursor once per tick (ids/timestamps recorded in _process_one).
        self.state.save()
        return processed

    async def _list_new_traces(self, from_ts: str, to_ts: str) -> list[dict[str, Any]]:
        from openweave_core.parser.trace_fetcher import list_recent_traces

        cfg = self.config
        collected: list[dict[str, Any]] = []
        page = 1
        while len(collected) < cfg.max_traces_per_tick:
            batch = await asyncio.to_thread(
                list_recent_traces,
                from_timestamp=from_ts,
                to_timestamp=to_ts,
                limit=cfg.page_limit,
                page=page,
                tags=cfg.tags,
                environment=cfg.environment,
                user_id=cfg.user_id,
                session_id=cfg.session_id,
                order_by="timestamp.asc",
            )
            if not batch:
                break
            for summary in batch:
                tid = str(summary.get("id") or "")
                if tid and not self.state.seen(tid):
                    collected.append(summary)
            if len(batch) < cfg.page_limit:
                break
            page += 1
        return collected

    async def _process_one(self, summary: dict[str, Any]) -> bool:
        trace_id = str(summary.get("id") or "")
        timestamp = summary.get("timestamp")
        if not trace_id:
            return False

        async with self._sem:
            from openweave_core.sidecar.assemble import assemble_analysis

            try:
                payload = await assemble_analysis(
                    trace_id, run_eval=self.config.run_eval
                )
            except (KeyError, ValueError) as exc:
                # Unparseable / not-found: record it so we don't loop on it.
                log.warning("skip trace %s: %s", trace_id, exc)
                self.state.mark_processed(trace_id, timestamp)
                return False
            except Exception as exc:  # noqa: BLE001
                self.stats.errors += 1
                self.stats.last_error = str(exc)
                log.exception("analyze failed for %s", trace_id)
                # Mark processed to avoid hot-looping a permanently bad trace.
                self.state.mark_processed(trace_id, timestamp)
                return False

        if self.config.trpc_project_id:
            payload.setdefault("projectId", self.config.trpc_project_id)

        self.sink.emit(trace_id, payload)
        self.state.mark_processed(trace_id, timestamp)
        self.stats.analyzed += 1
        if payload.get("flagged"):
            self.stats.flagged += 1
        return True


async def _amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    observer = TraceObserver()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, observer.request_stop)
        except NotImplementedError:
            # Windows: add_signal_handler isn't supported for SIGTERM.
            signal.signal(sig, lambda *_: observer.request_stop())

    await observer.run_forever()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
