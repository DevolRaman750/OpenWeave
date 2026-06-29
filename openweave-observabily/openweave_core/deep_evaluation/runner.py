"""
runner.py — Async batching runner with deterministic sampling.

Responsibilities
----------------
1. Take an ``IncidentReport`` (or its ``evaluation_plan``).
2. Filter tasks by category — only the 4 supported buckets.
3. Apply the sampling policy (priority-driven, deterministic per task).
4. Build a single shared ``ClaudeJudge`` and dispatch each surviving task
   via the appropriate ``MetricEvaluator``.
5. Run all tasks concurrently with bounded parallelism.
6. Wrap each call in failure isolation — one judge error never sinks the batch.
7. Return a ``BatchEvaluationResult`` with per-incident grouped outcomes.

Sampling policy (default)
-------------------------
    CRITICAL: 1.0    (always run)
    RISK:     0.5
    WARNING:  0.1
    INFO:     0.0    (always skip)

Sampling is **deterministic**: same task id always lands on the same side
of the threshold, so reruns produce comparable signal. Backed by a
SHA-256 over ``(incident_id, category)``.

Concurrency
-----------
We bound concurrent judge calls via a semaphore (default 8). Anthropic's
API rate-limits per minute; tune via ``OPENWEAVE_JUDGE_CONCURRENCY``.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from typing import Optional

from openweave_core.deep_evaluation.contracts import (
    BatchEvaluationResult,
    EvaluationCategory,
    EvaluationOutcome,
    EvaluationTask,
    IncidentEvaluation,
)
from openweave_core.deep_evaluation.judge import ClaudeJudge
from openweave_core.deep_evaluation.metric_registry import (
    MetricEvaluator,
    SUPPORTED_CATEGORIES,
    default_registry,
)
from openweave_core.incident_classification.contracts import (
    Incident,
    IncidentReport,
)
from openweave_core.sentinel_agent.findings import Severity


# ---------------------------------------------------------------------------
# Sampling policy
# ---------------------------------------------------------------------------

DEFAULT_SAMPLING_POLICY: dict[Severity, float] = {
    Severity.CRITICAL: 1.0,
    Severity.RISK: 0.5,
    Severity.WARNING: 0.1,
    Severity.INFO: 0.0,
}


def _sampling_bucket(task: EvaluationTask) -> float:
    key = f"{task.incident_id}|{task.category.value}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    val = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return val


def _should_sample(
    task: EvaluationTask, policy: dict[Severity, float]
) -> tuple[bool, str]:
    rate = policy.get(task.priority, 0.0)
    if rate >= 1.0:
        return True, ""
    if rate <= 0.0:
        return False, f"priority={task.priority.value} excluded by policy (rate=0)"
    bucket = _sampling_bucket(task)
    if bucket < rate:
        return True, ""
    return False, (
        f"priority={task.priority.value} sampled out "
        f"(bucket={bucket:.3f} >= rate={rate:.2f})"
    )


def _get_concurrency() -> int:
    try:
        return max(1, int(os.environ.get("OPENWEAVE_JUDGE_CONCURRENCY", "8")))
    except ValueError:
        return 8


# ---------------------------------------------------------------------------
# Core async API
# ---------------------------------------------------------------------------

async def evaluate_incident_report_async(
    report: IncidentReport,
    *,
    judge: Optional[ClaudeJudge] = None,
    registry: Optional[dict[EvaluationCategory, MetricEvaluator]] = None,
    sampling_policy: Optional[dict[Severity, float]] = None,
    concurrency: Optional[int] = None,
) -> BatchEvaluationResult:
    """Evaluate an IncidentReport's plan with DeepEval. Async-first."""
    judge = judge or ClaudeJudge()
    registry = registry or default_registry()
    policy = sampling_policy or DEFAULT_SAMPLING_POLICY
    sem = asyncio.Semaphore(concurrency or _get_concurrency())

    result = BatchEvaluationResult(
        trace_id=report.trace_id,
        judge_model=judge.get_model_name(),
        sampling_policy={k.value: v for k, v in policy.items()},
    )
    started = time.perf_counter()

    # Build per-task coroutine list, recording skips up front.
    runnable: list[tuple[EvaluationTask, MetricEvaluator]] = []
    skipped: list[EvaluationOutcome] = []

    for task in report.evaluation_plan:
        result.n_tasks_total += 1

        if task.category not in SUPPORTED_CATEGORIES or task.category not in registry:
            result.n_tasks_skipped += 1
            skipped.append(
                _skipped_outcome(
                    task,
                    f"category={task.category.value} not in supported set",
                )
            )
            continue

        sampled, reason = _should_sample(task, policy)
        if not sampled:
            result.n_tasks_skipped += 1
            skipped.append(_skipped_outcome(task, reason))
            continue

        evaluator = registry[task.category]
        runnable.append((task, evaluator))

    async def _bounded_eval(task: EvaluationTask, ev: MetricEvaluator):
        async with sem:
            return await ev.evaluate(task, judge)

    coroutines = [_bounded_eval(t, ev) for t, ev in runnable]
    outcomes: list[EvaluationOutcome] = []
    if coroutines:
        outcomes = await asyncio.gather(*coroutines, return_exceptions=False)
    result.n_tasks_sampled = len(outcomes)
    result.n_tasks_errored = sum(1 for o in outcomes if o.error)

    # Group all outcomes (run + skipped) by incident, preserving incident order
    # from the original report so consumers can join with report.incidents.
    grouped: dict[str, IncidentEvaluation] = {}
    sev_by_id: dict[str, Severity] = {
        inc.id: inc.severity for inc in report.incidents
    }
    for outcome in outcomes + skipped:
        ev = grouped.get(outcome.incident_id)
        if ev is None:
            ev = IncidentEvaluation(
                incident_id=outcome.incident_id,
                severity=sev_by_id.get(outcome.incident_id, Severity.INFO),
            )
            grouped[outcome.incident_id] = ev
        ev.outcomes.append(outcome)

    incident_order = [inc.id for inc in report.incidents]
    ordered = [grouped[i] for i in incident_order if i in grouped]
    # Append any outcomes whose incident_id wasn't in the report (defensive).
    for incident_id, ev in grouped.items():
        if incident_id not in incident_order:
            ordered.append(ev)
    result.evaluations = ordered

    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result


# ---------------------------------------------------------------------------
# Sync convenience
# ---------------------------------------------------------------------------

def evaluate_incident_report(
    report: IncidentReport,
    *,
    judge: Optional[ClaudeJudge] = None,
    registry: Optional[dict[EvaluationCategory, MetricEvaluator]] = None,
    sampling_policy: Optional[dict[Severity, float]] = None,
    concurrency: Optional[int] = None,
) -> BatchEvaluationResult:
    """Sync wrapper for environments that can't await."""
    return asyncio.run(
        evaluate_incident_report_async(
            report,
            judge=judge,
            registry=registry,
            sampling_policy=sampling_policy,
            concurrency=concurrency,
        )
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _skipped_outcome(task: EvaluationTask, reason: str) -> EvaluationOutcome:
    return EvaluationOutcome(
        incident_id=task.incident_id,
        category=task.category,
        metric_name="(skipped)",
        skipped_reason=reason,
    )


__all__ = [
    "evaluate_incident_report",
    "evaluate_incident_report_async",
    "DEFAULT_SAMPLING_POLICY",
]
