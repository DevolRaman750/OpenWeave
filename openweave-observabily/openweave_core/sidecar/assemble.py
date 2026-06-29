"""
assemble.py — Orchestrate the full pipeline and assemble one AnalysisPayload.

Stages (per the approved plan):
  1. run_anomaly_pipeline_full  -> (TraceAnomalyReport, TraceBundle)
  2. classify_trace             -> IncidentReport
  3. evaluate_incident_report   -> BatchEvaluationResult   (live Claude judge)
  4. build_graph                -> abstracted graph + propagation paths
  5. map everything             -> AnalysisPayload (mirrors AnalysisSeed)

The payload is plain JSON-serialisable dicts; the Node side validates it with
zod and writes it through Prisma. Nothing here touches a database.
"""

from __future__ import annotations

from typing import Any, Optional

from openweave_core.anomaly_pipeline.pipeline import (
    run_anomaly_pipeline_full_async,
)
from openweave_core.deep_evaluation.runner import evaluate_incident_report_async
from openweave_core.incident_classification.api import classify_trace
from openweave_core.sidecar import mapping
from openweave_core.sidecar.graph_build import build_graph


async def assemble_analysis(
    trace_id: str,
    *,
    run_eval: bool = True,
) -> dict[str, Any]:
    """Run the pipeline for *trace_id* and return the AnalysisPayload dict."""
    # 1. anomaly detection (parse + fan-out detectors)
    report, bundle = await run_anomaly_pipeline_full_async(trace_id)

    # 2. incident classification (heuristic, no LLM)
    incident_report = classify_trace(report, bundle)

    # 3. deep evaluation (live Claude judge; failure-isolated by the runner)
    batch = None
    if run_eval and incident_report.incidents:
        batch = await evaluate_incident_report_async(incident_report)

    # 4. abstracted graph + alignment context
    graph_payload, ctx = build_graph(bundle)
    propagation_paths = _propagation_paths(report)

    # 5. map -> payload
    eval_index = mapping.index_evaluations(batch)
    incidents_out: list[dict[str, Any]] = []
    flag_total = 0
    for incident in incident_report.incidents:
        member_flags = [ef.flag for ef in incident.flags]
        flag_total += len(member_flags)
        incidents_out.append(
            mapping.map_incident(
                incident,
                flags=member_flags,
                evaluation=eval_index.get(incident.id),
                ctx=ctx,
            )
        )

    overall = mapping.severity_name(incident_report.overall_severity.value)

    return {
        "traceId": trace_id,
        "overallSeverity": overall,
        "flagged": bool(report.flagged),
        "flagCount": flag_total,
        "incidentCount": len(incidents_out),
        "bySource": mapping.by_source(report),
        "categorySummary": {
            c.value: n for c, n in incident_report.category_summary.items()
        },
        "sourceSummary": dict(incident_report.source_summary),
        "envelope": _envelope(incident_report, report),
        "graph": graph_payload,
        "propagationPaths": propagation_paths,
        "evalJudgeModel": batch.judge_model if batch else "",
        "evalCounts": mapping.eval_counts(batch),
        "detectorResults": [
            mapping.map_detector_result(r) for r in report.detector_results
        ],
        "metadata": _metadata(report, batch),
        "parseDurationSeconds": float(report.parse_duration_seconds),
        "totalDurationSeconds": float(report.total_duration_seconds),
        "incidents": incidents_out,
    }


def assemble_analysis_sync(trace_id: str, *, run_eval: bool = True) -> dict[str, Any]:
    """Blocking wrapper for CLI / scripts (no running event loop)."""
    import asyncio

    return asyncio.run(assemble_analysis(trace_id, run_eval=run_eval))


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _propagation_paths(report) -> Optional[list[list[str]]]:
    """Collect sentinel propagation paths (already node-id sequences)."""
    paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for result in report.detector_results:
        for path in result.raw_summary.get("propagation_paths", []) or []:
            tup = tuple(str(x) for x in path)
            if len(tup) >= 2 and tup not in seen:
                seen.add(tup)
                paths.append(list(tup))
    return paths or None


def _envelope(incident_report, report) -> dict[str, Any]:
    env = incident_report.envelope
    if env is None:
        return {"input": "", "output": "", "tags": []}
    return {
        "input": env.input_text,
        "output": env.output_text,
        "name": env.name,
        "sessionId": env.session_id,
        "userId": env.user_id,
        "tags": list(env.tags),
        "release": env.release,
        "environment": env.environment,
        "metadata": mapping._jsonable(env.metadata),
    }


def _metadata(report, batch) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "n_spans": report.metadata.get("n_spans", len(report.flags)),
        "failed_detectors": report.failed_detectors(),
    }
    # carry through pipeline metadata that's JSON-safe
    for key in ("system_spec_source", "n_spans"):
        if key in report.metadata:
            meta[key] = mapping._jsonable(report.metadata[key])
    if batch is not None and batch.sampling_policy:
        meta["samplingPolicy"] = {
            mapping.severity_name(str(k)): v
            for k, v in batch.sampling_policy.items()
        }
    return meta
