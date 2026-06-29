"""processing_queue.py — bounded async work queue for trace analysis.

Why this exists
---------------
``run_anomaly_pipeline_*`` analyses one trace at a time and, when called
synchronously, blocks the caller for the whole duration. Under production load
that means a single trace stuck on a slow upstream (the embeddings endpoint)
stalls everything queued behind it.

This module decouples *submission* from *execution*. Traces are put on an
``asyncio.Queue``; a fixed pool of worker tasks pulls and processes them with
bounded concurrency. A slow trace occupies one worker — the other workers keep
draining the queue, so one slow upstream delays *that trace*, not the system.

This is the "restaurant kitchen" model: orders go on a ticket rail, a fixed
number of cooks work tickets in parallel, and one slow dish doesn't freeze the
whole dining room.

Layering
--------
This sits on top of the per-trace resilience already in place:
  * per-call timeout + bounded retries on the external clients,
  * a circuit breaker that fails fast when an endpoint is down,
  * a per-detector wall-clock cap in the fan-out.
The queue adds cross-trace isolation and back-pressure on top of those.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from openweave_core.models.span import ParsedSpan
from openweave_core.models.trace_envelope import TraceEnvelope
from openweave_core.sentinel_agent import SystemSpec

from openweave_core.anomaly_pipeline.contracts import TraceAnomalyReport
from openweave_core.anomaly_pipeline.pipeline import (
    run_anomaly_pipeline_for_spans_async,
)


@dataclass
class _Job:
    spans: list[ParsedSpan]
    trace_id: str
    system_spec: Optional[SystemSpec]
    envelope: Optional[TraceEnvelope]
    future: "asyncio.Future[TraceAnomalyReport]"


@dataclass
class QueueStats:
    """Lightweight counters for health endpoints / diagnostics."""
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    workers: int = 0
    pending: int = 0


class TraceProcessingQueue:
    """Bounded worker pool that analyses submitted traces concurrently.

    Parameters
    ----------
    workers     : number of concurrent worker tasks (max traces in flight).
    max_pending : queue capacity; 0 = unbounded. When full, ``submit_*``
                  awaits a free slot, applying natural back-pressure.

    Usage
    -----
        async with TraceProcessingQueue(workers=4) as q:
            report = await q.submit_spans(spans, trace_id="t1")

    or, fire many and gather:

        async with TraceProcessingQueue(workers=4) as q:
            reports = await asyncio.gather(*[
                q.submit_spans(s, trace_id=t) for s, t in jobs
            ])
    """

    def __init__(self, *, workers: int = 4, max_pending: int = 0) -> None:
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        self._n_workers = workers
        self._queue: "asyncio.Queue[_Job]" = asyncio.Queue(maxsize=max_pending)
        self._worker_tasks: list[asyncio.Task] = []
        self._started = False
        self._stats = QueueStats(workers=workers)

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> "TraceProcessingQueue":
        if not self._started:
            self._worker_tasks = [
                asyncio.create_task(self._worker(i), name=f"trace-worker-{i}")
                for i in range(self._n_workers)
            ]
            self._started = True
        return self

    async def aclose(self) -> None:
        """Cancel workers and drain. Pending unstarted jobs get CancelledError."""
        for t in self._worker_tasks:
            t.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []
        self._started = False
        # Fail any jobs still queued so awaiting callers don't hang forever.
        while not self._queue.empty():
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.cancel()
            self._queue.task_done()

    async def __aenter__(self) -> "TraceProcessingQueue":
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- submission ------------------------------------------------------

    async def submit_spans(
        self,
        spans: list[ParsedSpan],
        *,
        trace_id: str,
        system_spec: Optional[SystemSpec] = None,
        envelope: Optional[TraceEnvelope] = None,
    ) -> TraceAnomalyReport:
        """Enqueue a trace and await its report.

        Returns when a worker has finished this trace. Other submissions are
        processed concurrently up to ``workers``; the rest wait in the queue.
        """
        if not self._started:
            await self.start()
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[TraceAnomalyReport]" = loop.create_future()
        job = _Job(
            spans=spans,
            trace_id=trace_id,
            system_spec=system_spec,
            envelope=envelope,
            future=future,
        )
        await self._queue.put(job)
        self._stats.submitted += 1
        return await future

    # -- stats -----------------------------------------------------------

    def stats(self) -> QueueStats:
        self._stats.pending = self._queue.qsize()
        return self._stats

    # -- internals -------------------------------------------------------

    async def _worker(self, idx: int) -> None:
        while True:
            job = await self._queue.get()
            try:
                report = await run_anomaly_pipeline_for_spans_async(
                    job.spans,
                    trace_id=job.trace_id,
                    system_spec=job.system_spec,
                    envelope=job.envelope,
                )
                if not job.future.done():
                    job.future.set_result(report)
                self._stats.completed += 1
            except asyncio.CancelledError:
                # Shutting down: hand the job back as cancelled and exit.
                if not job.future.done():
                    job.future.cancel()
                self._queue.task_done()
                raise
            except BaseException as exc:  # noqa: BLE001 — never let a worker die
                if not job.future.done():
                    job.future.set_exception(exc)
                self._stats.failed += 1
            finally:
                # task_done for the normal/exception path (cancel path did it).
                if not job.future.cancelled():
                    self._queue.task_done()


# ---------------------------------------------------------------------------
# Sync convenience — process a batch of traces with bounded concurrency
# ---------------------------------------------------------------------------

def process_spans_batch(
    jobs: list[tuple[list[ParsedSpan], str]],
    *,
    workers: int = 4,
    system_spec: Optional[SystemSpec] = None,
) -> list[TraceAnomalyReport]:
    """Analyse many ``(spans, trace_id)`` jobs concurrently via a worker pool.

    Synchronous facade for scripts/batch jobs: spins up a queue, submits all
    jobs, and returns reports in input order. At most ``workers`` traces run at
    once; a slow trace only holds up its own worker.
    """
    async def _run() -> list[TraceAnomalyReport]:
        async with TraceProcessingQueue(workers=workers) as q:
            return await asyncio.gather(*[
                q.submit_spans(spans, trace_id=tid, system_spec=system_spec)
                for spans, tid in jobs
            ])

    return asyncio.run(_run())
