"""
contracts.py — Types for the DeepEval wrapper layer.

Three concrete types:

    EvaluationOutcome    : one metric result for one task.
    IncidentEvaluation   : all outcomes for one incident, aggregated.
    BatchEvaluationResult: full run output across a whole IncidentReport.

The outcomes are framework-agnostic — they look the same whether the
metric came from DeepEval, Ragas, Patronus or a custom protocol impl.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from openweave_core.incident_classification.contracts import (
    EvaluationCategory,
    EvaluationTask,
)
from openweave_core.sentinel_agent.findings import Severity


@dataclass
class EvaluationOutcome:
    """One metric run for one (incident, category) task."""

    incident_id: str
    category: EvaluationCategory
    metric_name: str
    score: float = 0.0
    threshold: float = 0.0
    passed: bool = False
    reason: str = ""
    judge_model: str = ""
    duration_ms: float = 0.0
    error: Optional[str] = None
    skipped_reason: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.skipped_reason is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "category": self.category.value,
            "metric_name": self.metric_name,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "reason": self.reason,
            "judge_model": self.judge_model,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
            "extras": self.extras,
        }


@dataclass
class IncidentEvaluation:
    """All outcomes for a single incident."""

    incident_id: str
    severity: Severity
    outcomes: list[EvaluationOutcome] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(o.passed for o in self.outcomes if o.succeeded)

    @property
    def lowest_score(self) -> float:
        scored = [o.score for o in self.outcomes if o.succeeded]
        return min(scored) if scored else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "severity": self.severity.value,
            "all_passed": self.all_passed,
            "lowest_score": self.lowest_score,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


@dataclass
class BatchEvaluationResult:
    """Top-level run output for one IncidentReport."""

    trace_id: str
    evaluations: list[IncidentEvaluation] = field(default_factory=list)
    n_tasks_total: int = 0
    n_tasks_sampled: int = 0
    n_tasks_skipped: int = 0
    n_tasks_errored: int = 0
    judge_model: str = ""
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_ms: float = 0.0
    sampling_policy: dict[str, float] = field(default_factory=dict)

    def outcomes(self) -> list[EvaluationOutcome]:
        return [o for e in self.evaluations for o in e.outcomes]

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "judge_model": self.judge_model,
            "started_at": self.started_at.isoformat(),
            "duration_ms": self.duration_ms,
            "n_tasks_total": self.n_tasks_total,
            "n_tasks_sampled": self.n_tasks_sampled,
            "n_tasks_skipped": self.n_tasks_skipped,
            "n_tasks_errored": self.n_tasks_errored,
            "sampling_policy": dict(self.sampling_policy),
            "evaluations": [e.as_dict() for e in self.evaluations],
        }


# Re-exports so callers only need to import from this module.
__all__ = [
    "EvaluationOutcome",
    "IncidentEvaluation",
    "BatchEvaluationResult",
    "EvaluationTask",
    "EvaluationCategory",
]
