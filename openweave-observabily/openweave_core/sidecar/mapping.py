"""
mapping.py — Pure normalizers: contract objects -> AnalysisPayload JSON fields.

The Prisma enums surface UPPERCASE member *names* (``CRITICAL``,
``TOOL_INVOCATION``) while the Python contracts carry lowercase ``.value``s
(``"critical"``, ``"tool_invocation"``). Everything that lands in an enum
column is normalised here. Flag/incident *subject ids* are translated onto graph
nodes/edges via :func:`graph_subject_id` so the Report, Graph and Eval views all
address the same elements.
"""

from __future__ import annotations

from typing import Any, Optional

from openweave_core.anomaly_pipeline.contracts import (
    DetectorResult,
    NormalizedFlag,
    TraceAnomalyReport,
)
from openweave_core.deep_evaluation.contracts import (
    BatchEvaluationResult,
    EvaluationOutcome,
    IncidentEvaluation,
)
from openweave_core.incident_classification.contracts import Incident
from openweave_core.sidecar.graph_build import GraphContext, graph_subject_id

_SEVERITY_NAMES = {"info", "warning", "risk", "critical"}
_CATEGORY_NAMES = {
    "prompt",
    "tool_invocation",
    "llm_generation",
    "rag",
    "observability",
    "safety",
}


def severity_name(value: str) -> str:
    """``Severity.value`` ("critical") -> OwSeverity member name ("CRITICAL")."""
    v = (value or "").lower()
    return v.upper() if v in _SEVERITY_NAMES else "INFO"


def category_name(value: str) -> str:
    """``EvaluationCategory.value`` -> OwEvaluationCategory member name."""
    v = (value or "").lower()
    return v.upper() if v in _CATEGORY_NAMES else "OBSERVABILITY"


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def map_flag(flag: NormalizedFlag, ctx: GraphContext) -> dict[str, Any]:
    return {
        "flagKey": flag.id,
        "sourcePipeline": flag.source_pipeline,
        "category": flag.category,  # free-form detector code (String column)
        "severity": severity_name(flag.severity.value),
        "subjectId": graph_subject_id(flag.subject_id, flag.subject_type, ctx),
        "subjectType": flag.subject_type,
        "confidence": float(flag.confidence),
        "message": flag.message,
        "evidence": _jsonable(flag.evidence),
    }


def map_detector_result(result: DetectorResult) -> dict[str, Any]:
    return {
        "detector": result.detector,
        "ok": bool(result.ok),
        "flagCount": result.flag_count,
        "durationSeconds": float(result.duration_seconds),
        "error": result.error,
        "warning": result.warning,
        "rawSummary": _jsonable(result.raw_summary),
    }


# ---------------------------------------------------------------------------
# Evaluation outcomes
# ---------------------------------------------------------------------------

def map_outcome(outcome: EvaluationOutcome) -> dict[str, Any]:
    return {
        "category": category_name(outcome.category.value),
        "metricName": outcome.metric_name,
        "score": float(outcome.score),
        "threshold": float(outcome.threshold),
        "passed": bool(outcome.passed),
        "reason": outcome.reason or "",
        "judgeModel": outcome.judge_model or "",
        "durationMs": float(outcome.duration_ms),
        "error": outcome.error,
        "skippedReason": outcome.skipped_reason,
        "extras": _jsonable(outcome.extras),
    }


# ---------------------------------------------------------------------------
# Incidents (+ joined flags & eval outcomes)
# ---------------------------------------------------------------------------

def map_incident(
    incident: Incident,
    *,
    flags: list[NormalizedFlag],
    evaluation: Optional[IncidentEvaluation],
    ctx: GraphContext,
) -> dict[str, Any]:
    member_flags = [map_flag(f, ctx) for f in flags]
    outcomes = (
        [map_outcome(o) for o in evaluation.outcomes] if evaluation else []
    )

    return {
        "incidentKey": incident.id,
        "title": incident.title,
        "description": incident.description,
        "severity": severity_name(incident.severity.value),
        "confidence": float(incident.confidence),
        "primarySubjectId": _subject(incident.primary_subject_id, flags, ctx),
        "subjectIds": _subjects(sorted(incident.subject_ids), flags, ctx),
        "affectedSpanIds": sorted(incident.affected_span_ids),
        "sourcePipelines": sorted(incident.source_pipelines),
        "categories": [category_name(c.value) for c in incident.categories],
        "categoryEvidence": {
            c.value: list(ev) for c, ev in incident.category_evidence.items()
        },
        "correlationSignals": list(incident.correlation_signals),
        "evalAllPassed": evaluation.all_passed if evaluation else True,
        "evalLowestScore": evaluation.lowest_score if evaluation else 0.0,
        "flags": member_flags,
        "outcomes": outcomes,
    }


# ---------------------------------------------------------------------------
# Eval-batch index helpers
# ---------------------------------------------------------------------------

def index_evaluations(
    batch: Optional[BatchEvaluationResult],
) -> dict[str, IncidentEvaluation]:
    if batch is None:
        return {}
    return {e.incident_id: e for e in batch.evaluations}


def eval_counts(batch: Optional[BatchEvaluationResult]) -> dict[str, int]:
    if batch is None:
        return {
            "n_tasks_total": 0,
            "n_tasks_sampled": 0,
            "n_tasks_skipped": 0,
            "n_tasks_errored": 0,
        }
    return {
        "n_tasks_total": batch.n_tasks_total,
        "n_tasks_sampled": batch.n_tasks_sampled,
        "n_tasks_skipped": batch.n_tasks_skipped,
        "n_tasks_errored": batch.n_tasks_errored,
    }


def by_source(report: TraceAnomalyReport) -> dict[str, int]:
    return dict(report.by_source)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _subject_lookup(flags: list[NormalizedFlag]) -> dict[str, str]:
    """raw subject_id -> subject_type, from this incident's member flags."""
    out: dict[str, str] = {}
    for f in flags:
        out.setdefault(f.subject_id, f.subject_type)
        # span_pair flags expose each half as a span subject too.
        if f.subject_type == "span_pair":
            for half in f.subject_id.split("::"):
                out.setdefault(half, "span")
    return out


def _subject(raw: str, flags: list[NormalizedFlag], ctx: GraphContext) -> str:
    if not raw:
        return ""
    st = _subject_lookup(flags).get(raw)
    if st is None:
        # Best-effort: a known span id resolves to its node, else node:<id>.
        st = "span" if raw in ctx.span_to_node else "node"
    return graph_subject_id(raw, st, ctx)


def _subjects(
    raws: list[str], flags: list[NormalizedFlag], ctx: GraphContext
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in raws:
        mapped = _subject(raw, flags, ctx)
        if mapped and mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out


def _jsonable(value: Any) -> Any:
    """Best-effort coercion so detector evidence survives JSON serialisation."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
