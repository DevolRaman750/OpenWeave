"""
pipeline.py — Single public entry point for the unified anomaly pipeline.

Flow
----
    trace_id  →  parser (once)  →  TraceBundle
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                   adaptive_baseline  sentinel    cycle_detection
                          └───────────┼───────────┘
                                      │
                                normalised flags
                                      │
                              TraceAnomalyReport

Concurrency
-----------
Detectors are CPU-bound Python code with occasional I/O (the cycle detector
optionally hits the NVIDIA embeddings endpoint). We use ``asyncio`` with
``asyncio.to_thread`` so each detector runs in its own worker thread, GIL
permitting they overlap on I/O waits and on numpy/blas work that releases
the GIL. The sync entry point wraps the async coroutine via ``asyncio.run``.

Failure isolation
-----------------
``asyncio.gather(..., return_exceptions=True)`` ensures one detector raising
does not cancel the others. Each detector exception becomes a failed
``DetectorResult`` carrying the error string; the report still aggregates
successful detectors' flags.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from typing import Any, Awaitable, Callable, Optional

from openweave_core.models.span import ParsedSpan
from openweave_core.models.trace_envelope import TraceEnvelope
from openweave_core.sentinel_agent import Severity, SystemSpec

from openweave_core.anomaly_pipeline.contracts import (
    ALL_SOURCES,
    DetectorResult,
    NormalizedFlag,
    SOURCE_ADAPTIVE_BASELINE,
    SOURCE_CYCLE_DETECTION,
    SOURCE_SENTINEL_AGENT,
    TraceAnomalyReport,
    TraceBundle,
)
from openweave_core.anomaly_pipeline.runners import (
    run_adaptive_baseline,
    run_cycle_detection,
    run_sentinel_agent,
)


# ---------------------------------------------------------------------------
# Parser injection — default = the live Langfuse parser
# ---------------------------------------------------------------------------

# Span-only parser: backwards-compatible shape, ``trace_id → list[ParsedSpan]``.
# When supplied alone, the pipeline synthesises an empty TraceEnvelope so the
# bundle shape stays uniform for downstream layers.
Parser = Callable[[str], list[ParsedSpan]]

# Full parser: returns the trace envelope alongside the spans in a single
# fetch. Use this form when you want trace-level input/output/tags available
# to the classifier / RAG router.
FullParser = Callable[[str], tuple[TraceEnvelope, list[ParsedSpan]]]

# Type for a per-detector runner exception or success.
RunnerFn = Callable[..., DetectorResult]

# Hard per-detector wall-clock cap. A detector that blocks on a slow upstream
# (e.g. the embeddings endpoint) is cut off here so it can't stall the whole
# fan-out; it becomes an ok=False DetectorResult and the others still report.
# This backstops the client-level timeout + circuit breaker (which usually fire
# first and degrade gracefully). Set generous; tune via env.
DEFAULT_DETECTOR_TIMEOUT_SECONDS: float = float(
    os.environ.get("OPENWEAVE_DETECTOR_TIMEOUT", "20")
)


def _default_full_parser() -> FullParser:
    """Late-import the live full parser so importing this module is side-effect free."""
    from openweave_core.parser import parse_langfuse_trace_full
    return parse_langfuse_trace_full


def _wrap_span_only_parser(parser: Parser) -> FullParser:
    """Adapt a span-only parser to the FullParser signature with an empty envelope."""

    def _wrapped(trace_id: str) -> tuple[TraceEnvelope, list[ParsedSpan]]:
        spans = parser(trace_id)
        return TraceEnvelope.empty(trace_id), spans

    return _wrapped


# ---------------------------------------------------------------------------
# Concurrency helpers
# ---------------------------------------------------------------------------

async def _run_detector(
    detector_name: str,
    runner: RunnerFn,
    *runner_args: Any,
    _timeout: Optional[float] = None,
    **runner_kwargs: Any,
) -> DetectorResult:
    """Run *runner* in a worker thread; never raises — errors become DetectorResult.ok=False.

    If ``_timeout`` is set and the runner exceeds it, this returns a timed-out
    ``ok=False`` result so a single slow detector cannot stall the fan-out.
    (The worker thread itself is not killable; it unwinds on its own once the
    client-level timeout/breaker fire — its late result is simply discarded.)
    """
    start = time.perf_counter()
    try:
        call = asyncio.to_thread(runner, *runner_args, **runner_kwargs)
        if _timeout is not None:
            result: DetectorResult = await asyncio.wait_for(call, _timeout)
        else:
            result = await call
        # Defensive: ensure the runner returned the right type.
        if not isinstance(result, DetectorResult):
            raise TypeError(
                f"{detector_name} runner returned {type(result).__name__}, "
                "expected DetectorResult"
            )
        result.duration_seconds = time.perf_counter() - start
        return result
    except (asyncio.TimeoutError, TimeoutError):
        return DetectorResult(
            detector=detector_name,
            ok=False,
            flags=[],
            duration_seconds=time.perf_counter() - start,
            error=f"timed out after {_timeout:g}s",
            raw_summary={"timeout_seconds": _timeout},
        )
    except BaseException as exc:  # noqa: BLE001 — we want to capture *everything*
        return DetectorResult(
            detector=detector_name,
            ok=False,
            flags=[],
            duration_seconds=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
            raw_summary={"traceback": traceback.format_exc()},
        )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.RISK: 2,
    Severity.CRITICAL: 3,
}


def _max_severity(flags: list[NormalizedFlag]) -> Severity:
    if not flags:
        return Severity.INFO
    return max(
        (f.severity for f in flags),
        key=lambda s: _SEVERITY_RANK[s],
    )


def _build_report(
    bundle: TraceBundle,
    detector_results: list[DetectorResult],
    parse_duration: float,
    total_duration: float,
) -> TraceAnomalyReport:
    all_flags: list[NormalizedFlag] = []
    by_source: dict[str, int] = {s: 0 for s in ALL_SOURCES}
    for r in detector_results:
        # Init counter for any detector outside ALL_SOURCES too (future-proof).
        by_source.setdefault(r.detector, 0)
        by_source[r.detector] = r.flag_count
        all_flags.extend(r.flags)

    # Stable order: highest severity first, then by source, then by detected_at.
    all_flags.sort(
        key=lambda f: (
            -_SEVERITY_RANK[f.severity],
            f.source_pipeline,
            f.detected_at.isoformat(),
        )
    )

    return TraceAnomalyReport(
        trace_id=bundle.trace_id,
        flagged=bool(all_flags),
        flags=all_flags,
        by_source=by_source,
        detector_results=detector_results,
        overall_severity=_max_severity(all_flags),
        parse_duration_seconds=parse_duration,
        total_duration_seconds=total_duration,
        metadata={
            "n_spans": len(bundle.spans),
            "n_detectors_run": len(detector_results),
            "n_detectors_failed": sum(1 for r in detector_results if not r.ok),
        },
    )


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------

async def run_anomaly_pipeline_async(
    trace_id: str,
    *,
    system_spec: Optional[SystemSpec] = None,
    parser: Optional[Parser] = None,
    full_parser: Optional[FullParser] = None,
) -> TraceAnomalyReport:
    """Async variant — fetch + parse once, fan out the 3 detectors concurrently.

    Parameters
    ----------
    trace_id    : trace identifier accepted by the parser.
    system_spec : optional SystemSpec for sentinel's static graph. If None,
                  sentinel auto-derives a permissive spec from observed nodes.
    parser      : optional span-only parser (``Callable[[trace_id], list[ParsedSpan]]``).
                  When supplied, the envelope is synthesised as empty.
    full_parser : optional full parser returning ``(envelope, spans)``.
                  Defaults to ``parse_langfuse_trace_full`` when both are None.

    Returns
    -------
    TraceAnomalyReport
    """
    _report, _bundle = await _orchestrate_async(
        trace_id=trace_id,
        system_spec=system_spec,
        parser=parser,
        full_parser=full_parser,
    )
    return _report


async def run_anomaly_pipeline_full_async(
    trace_id: str,
    *,
    system_spec: Optional[SystemSpec] = None,
    parser: Optional[Parser] = None,
    full_parser: Optional[FullParser] = None,
) -> tuple[TraceAnomalyReport, TraceBundle]:
    """Async variant that also returns the ``TraceBundle`` — what the
    incident-classification layer consumes."""
    return await _orchestrate_async(
        trace_id=trace_id,
        system_spec=system_spec,
        parser=parser,
        full_parser=full_parser,
    )


async def _orchestrate_async(
    *,
    trace_id: str,
    system_spec: Optional[SystemSpec],
    parser: Optional[Parser],
    full_parser: Optional[FullParser],
) -> tuple[TraceAnomalyReport, TraceBundle]:
    if full_parser is None and parser is None:
        full_parser = _default_full_parser()
    elif full_parser is None:
        full_parser = _wrap_span_only_parser(parser)  # type: ignore[arg-type]

    overall_start = time.perf_counter()

    parse_start = time.perf_counter()
    try:
        envelope, spans = await asyncio.to_thread(full_parser, trace_id)
    except BaseException as exc:  # noqa: BLE001 — parser failure is fatal
        raise RuntimeError(
            f"trace parser failed for trace_id={trace_id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    parse_duration = time.perf_counter() - parse_start

    return await _fan_out(
        trace_id=trace_id,
        spans=spans,
        envelope=envelope,
        system_spec=system_spec,
        parse_duration=parse_duration,
        overall_start=overall_start,
    )


async def run_anomaly_pipeline_for_spans_async(
    spans: list[ParsedSpan],
    *,
    trace_id: str,
    system_spec: Optional[SystemSpec] = None,
    envelope: Optional[TraceEnvelope] = None,
) -> TraceAnomalyReport:
    """Variant that skips fetching — caller supplies already-parsed spans.

    ``envelope`` is optional; when omitted an empty envelope is constructed
    so the downstream contract stays uniform.
    """
    report, _bundle = await _fan_out(
        trace_id=trace_id,
        spans=list(spans),
        envelope=envelope or TraceEnvelope.empty(trace_id),
        system_spec=system_spec,
        parse_duration=0.0,
        overall_start=time.perf_counter(),
    )
    return report


async def run_anomaly_pipeline_for_spans_full_async(
    spans: list[ParsedSpan],
    *,
    trace_id: str,
    system_spec: Optional[SystemSpec] = None,
    envelope: Optional[TraceEnvelope] = None,
) -> tuple[TraceAnomalyReport, TraceBundle]:
    """``run_anomaly_pipeline_for_spans_async`` variant that returns the bundle."""
    return await _fan_out(
        trace_id=trace_id,
        spans=list(spans),
        envelope=envelope or TraceEnvelope.empty(trace_id),
        system_spec=system_spec,
        parse_duration=0.0,
        overall_start=time.perf_counter(),
    )


async def _fan_out(
    *,
    trace_id: str,
    spans: list[ParsedSpan],
    envelope: TraceEnvelope,
    system_spec: Optional[SystemSpec],
    parse_duration: float,
    overall_start: float,
    detector_timeout: Optional[float] = DEFAULT_DETECTOR_TIMEOUT_SECONDS,
) -> tuple[TraceAnomalyReport, TraceBundle]:
    bundle = TraceBundle.from_spans(
        trace_id=trace_id, spans=spans, envelope=envelope
    )

    detector_results = await asyncio.gather(
        _run_detector(SOURCE_ADAPTIVE_BASELINE, run_adaptive_baseline, bundle,
                      _timeout=detector_timeout),
        _run_detector(SOURCE_SENTINEL_AGENT, run_sentinel_agent, bundle,
                      _timeout=detector_timeout, system_spec=system_spec),
        _run_detector(SOURCE_CYCLE_DETECTION, run_cycle_detection, bundle,
                      _timeout=detector_timeout),
    )

    total_duration = time.perf_counter() - overall_start
    report = _build_report(
        bundle, list(detector_results), parse_duration, total_duration
    )
    return report, bundle


# ---------------------------------------------------------------------------
# Sync entry point (the recommended public API)
# ---------------------------------------------------------------------------

def run_anomaly_pipeline(
    trace_id: str,
    *,
    system_spec: Optional[SystemSpec] = None,
    parser: Optional[Parser] = None,
    full_parser: Optional[FullParser] = None,
) -> TraceAnomalyReport:
    """Synchronous one-shot entry point.

    Internally runs ``run_anomaly_pipeline_async`` via ``asyncio.run``.
    Use this from regular scripts; use the ``_async`` variant when you are
    already inside an event loop (FastAPI handler, Jupyter kernel, etc.).
    """
    return asyncio.run(
        run_anomaly_pipeline_async(
            trace_id,
            system_spec=system_spec,
            parser=parser,
            full_parser=full_parser,
        )
    )


def run_anomaly_pipeline_full(
    trace_id: str,
    *,
    system_spec: Optional[SystemSpec] = None,
    parser: Optional[Parser] = None,
    full_parser: Optional[FullParser] = None,
) -> tuple[TraceAnomalyReport, TraceBundle]:
    """Synchronous variant returning ``(report, bundle)`` — what the incident
    classification layer feeds on."""
    return asyncio.run(
        run_anomaly_pipeline_full_async(
            trace_id,
            system_spec=system_spec,
            parser=parser,
            full_parser=full_parser,
        )
    )


def run_anomaly_pipeline_for_spans(
    spans: list[ParsedSpan],
    *,
    trace_id: str,
    system_spec: Optional[SystemSpec] = None,
    envelope: Optional[TraceEnvelope] = None,
) -> TraceAnomalyReport:
    """Synchronous variant that skips the parser."""
    return asyncio.run(
        run_anomaly_pipeline_for_spans_async(
            spans, trace_id=trace_id, system_spec=system_spec, envelope=envelope,
        )
    )


def run_anomaly_pipeline_for_spans_full(
    spans: list[ParsedSpan],
    *,
    trace_id: str,
    system_spec: Optional[SystemSpec] = None,
    envelope: Optional[TraceEnvelope] = None,
) -> tuple[TraceAnomalyReport, TraceBundle]:
    """Sync ``_for_spans`` variant returning ``(report, bundle)``."""
    return asyncio.run(
        run_anomaly_pipeline_for_spans_full_async(
            spans, trace_id=trace_id, system_spec=system_spec, envelope=envelope,
        )
    )
