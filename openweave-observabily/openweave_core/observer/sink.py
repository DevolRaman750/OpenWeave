"""Sinks: where the observer sends a finished AnalysisPayload.

Sinks are intentionally simple and composable. The default deployment writes
one JSON report per trace to disk and logs a one-line summary — fully
self-contained, no Node/UI required. An optional tRPC sink forwards the payload
to the Langfuse-based OpenWeave UI writer when you want results in the web app.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Protocol

log = logging.getLogger("openweave.observer.sink")


class Sink(Protocol):
    def emit(self, trace_id: str, payload: dict[str, Any]) -> None: ...


class LogSink:
    """One structured line per analyzed trace — feeds container log scraping."""

    def emit(self, trace_id: str, payload: dict[str, Any]) -> None:
        log.info(
            "analyzed trace=%s severity=%s flagged=%s flags=%d incidents=%d "
            "spans=%s dur=%.3fs",
            trace_id,
            payload.get("overallSeverity"),
            payload.get("flagged"),
            int(payload.get("flagCount", 0) or 0),
            int(payload.get("incidentCount", 0) or 0),
            (payload.get("metadata") or {}).get("n_spans"),
            float(payload.get("totalDurationSeconds", 0.0) or 0.0),
        )


class FileSink:
    """Persist the full payload as ``<output_dir>/<trace_id>.json``."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, trace_id: str, payload: dict[str, Any]) -> None:
        safe = trace_id.replace("/", "_").replace("\\", "_")
        path = self.output_dir / f"{safe}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


class TrpcSink:
    """POST the payload to the Node/Langfuse OpenWeave writer (UI persistence).

    The endpoint is the batched tRPC mutation ``openweave.analyze.run``; it
    requires auth, so ``auth_header`` must carry a valid session cookie or token
    (scope ``evalJob:CUD``). Best-effort: failures are logged, not fatal — the
    FileSink copy is the durable record.
    """

    def __init__(
        self,
        url: str,
        project_id: str,
        auth_header: Optional[str] = None,
    ) -> None:
        self.url = url
        self.project_id = project_id
        self.auth_header = auth_header

    def emit(self, trace_id: str, payload: dict[str, Any]) -> None:
        body = json.dumps(
            {"0": {"json": {"projectId": self.project_id, "payload": payload}}}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.auth_header:
            name, _, value = self.auth_header.partition(":")
            headers[name.strip() or "Cookie"] = value.strip() or self.auth_header
        request = urllib.request.Request(self.url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            log.warning("trpc sink %s -> %s: %s", trace_id, exc.code, detail[:200])
        except Exception as exc:  # noqa: BLE001 — sink must never crash the loop
            log.warning("trpc sink %s failed: %s", trace_id, exc)


class CompositeSink:
    def __init__(self, sinks: list[Sink]) -> None:
        self.sinks = sinks

    def emit(self, trace_id: str, payload: dict[str, Any]) -> None:
        for sink in self.sinks:
            try:
                sink.emit(trace_id, payload)
            except Exception as exc:  # noqa: BLE001
                log.warning("sink %s failed for %s: %s", type(sink).__name__, trace_id, exc)


def build_sink(config) -> Sink:
    """Assemble the configured sinks into one composite."""
    sinks: list[Sink] = []
    if config.log_summaries:
        sinks.append(LogSink())
    if config.output_dir:
        sinks.append(FileSink(config.output_dir))
    if config.trpc_url and config.trpc_project_id:
        sinks.append(
            TrpcSink(config.trpc_url, config.trpc_project_id, config.trpc_auth_header)
        )
    return CompositeSink(sinks)
