"""
anomaly_pipeline — Unified anomaly-detection facade over the three engines.

One public function ``run_anomaly_pipeline(trace_id)`` fetches+parses the
trace once and concurrently runs:

    * adaptive_baseline (AMDM rolling z-score + EWMA + Mahalanobis)
    * sentinel_agent    (rule + judge + WL attack-path matcher)
    * cycle_detection   (CDCS + DAG + embedding-based semantic confirmation)

Every engine's native output is mapped into a uniform ``NormalizedFlag`` so
the downstream classification layer (RAG / Prompt / LLM-generation / Tool)
consumes a single contract, not three.

Usage
-----
    from openweave_core.anomaly_pipeline import run_anomaly_pipeline

    report = run_anomaly_pipeline("868319fa7d359241e9860271a6a92fbb")

    if report.flagged:
        for flag in report.flags:
            print(flag.source_pipeline, flag.severity.value,
                  flag.subject_id, flag.message)
"""

from openweave_core.anomaly_pipeline.contracts import (
    ALL_SOURCES,
    DetectorResult,
    NormalizedFlag,
    SOURCE_ADAPTIVE_BASELINE,
    SOURCE_CYCLE_DETECTION,
    SOURCE_SENTINEL_AGENT,
    SUBJECT_ATTACK_PATH,
    SUBJECT_EDGE,
    SUBJECT_NODE,
    SUBJECT_SPAN,
    SUBJECT_SPAN_PAIR,
    SUBJECT_TRACE,
    TraceAnomalyReport,
    TraceBundle,
)
from openweave_core.anomaly_pipeline.runners import (
    run_adaptive_baseline,
    run_cycle_detection,
    run_sentinel_agent,
)
from openweave_core.anomaly_pipeline.pipeline import (
    FullParser,
    Parser,
    run_anomaly_pipeline,
    run_anomaly_pipeline_async,
    run_anomaly_pipeline_for_spans,
    run_anomaly_pipeline_for_spans_async,
    run_anomaly_pipeline_for_spans_full,
    run_anomaly_pipeline_for_spans_full_async,
    run_anomaly_pipeline_full,
    run_anomaly_pipeline_full_async,
)
from openweave_core.anomaly_pipeline.processing_queue import (
    QueueStats,
    TraceProcessingQueue,
    process_spans_batch,
)

__all__ = [
    # Public entry points
    "run_anomaly_pipeline",
    "run_anomaly_pipeline_async",
    "run_anomaly_pipeline_full",
    "run_anomaly_pipeline_full_async",
    "run_anomaly_pipeline_for_spans",
    "run_anomaly_pipeline_for_spans_async",
    "run_anomaly_pipeline_for_spans_full",
    "run_anomaly_pipeline_for_spans_full_async",
    # Async/queue execution
    "TraceProcessingQueue",
    "QueueStats",
    "process_spans_batch",
    # Output types
    "TraceAnomalyReport",
    "NormalizedFlag",
    "DetectorResult",
    "TraceBundle",
    # Source / subject identifiers
    "SOURCE_ADAPTIVE_BASELINE",
    "SOURCE_SENTINEL_AGENT",
    "SOURCE_CYCLE_DETECTION",
    "ALL_SOURCES",
    "SUBJECT_SPAN",
    "SUBJECT_NODE",
    "SUBJECT_EDGE",
    "SUBJECT_ATTACK_PATH",
    "SUBJECT_SPAN_PAIR",
    "SUBJECT_TRACE",
    # Direct runner access (for callers building custom orchestrators)
    "run_adaptive_baseline",
    "run_sentinel_agent",
    "run_cycle_detection",
    # Type aliases
    "Parser",
    "FullParser",
]
