"""
test_deep_evaluation.py — Smoke tests for the DeepEval wrapper.

These tests do NOT call the live Claude API. They verify:

1. Package wiring — every public symbol importable.
2. Sampling — deterministic, follows the policy.
3. Case builder — each category builds a sensible LLMTestCase.
4. Registry — every supported category resolves to a MetricEvaluator.
5. Runner — with a mock judge + mock evaluator, the runner skips
   unsupported categories, applies the sampling policy, and groups
   outcomes by incident.
6. Failure isolation — an evaluator that raises doesn't sink the batch.

Run:
    python test_deep_evaluation.py
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

# Make openweave_core importable when running this script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openweave_core.deep_evaluation import (
    BatchEvaluationResult,
    DEFAULT_SAMPLING_POLICY,
    EvaluationCategory,
    EvaluationOutcome,
    EvaluationTask,
    MetricEvaluator,
    SUPPORTED_CATEGORIES,
    default_registry,
    evaluate_incident_report_async,
)
from openweave_core.deep_evaluation.case_builder import build_test_case
from openweave_core.deep_evaluation.runner import _should_sample
from openweave_core.incident_classification.contracts import (
    Incident,
    IncidentReport,
)
from openweave_core.sentinel_agent.findings import Severity


# ---------------------------------------------------------------------------
# Test infra
# ---------------------------------------------------------------------------

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def run_test(fn):
    name = fn.__name__
    try:
        result = fn()
        if asyncio.iscoroutine(result):
            asyncio.run(result)
        PASSED.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILED.append((name, str(e) or "assertion failed"))
        print(f"  FAIL  {name}: {e}")
    except Exception:
        FAILED.append((name, traceback.format_exc()))
        print(f"  FAIL  {name}:\n{traceback.format_exc()}")
    return fn


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockJudge:
    """Stand-in for ClaudeJudge — never calls a real API."""

    def get_model_name(self) -> str:
        return "mock-judge-v0"


class MockEvaluator:
    metric_name = "mock.PerfectScore"

    def __init__(self, *, raise_for: set[str] | None = None) -> None:
        self.calls: list[EvaluationTask] = []
        self.raise_for = raise_for or set()

    async def evaluate(self, task, judge):
        self.calls.append(task)
        outcome = EvaluationOutcome(
            incident_id=task.incident_id,
            category=task.category,
            metric_name=self.metric_name,
            threshold=0.5,
            judge_model=judge.get_model_name(),
        )
        if task.incident_id in self.raise_for:
            try:
                raise RuntimeError("synthetic judge failure")
            except RuntimeError as e:
                outcome.error = str(e)
            return outcome
        outcome.score = 0.9
        outcome.passed = True
        outcome.reason = "mock ok"
        return outcome


def _make_task(
    incident_id: str,
    category: EvaluationCategory,
    priority: Severity,
    context: dict | None = None,
) -> EvaluationTask:
    return EvaluationTask(
        incident_id=incident_id,
        category=category,
        priority=priority,
        subject_id=f"{incident_id}-subject",
        context=context or {"trace_input": "hello", "trace_output": "world"},
    )


def _make_report(*tasks: EvaluationTask) -> IncidentReport:
    incidents: dict[str, Incident] = {}
    for t in tasks:
        if t.incident_id not in incidents:
            incidents[t.incident_id] = Incident(
                id=t.incident_id,
                flags=[],
                severity=t.priority,
            )
    return IncidentReport(
        trace_id="trace-xyz",
        incidents=list(incidents.values()),
        evaluation_plan=list(tasks),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@run_test
def test_package_imports():
    from openweave_core.deep_evaluation import (
        ClaudeJudge,
        default_registry,
        evaluate_incident_report,
    )
    assert callable(default_registry)
    assert callable(evaluate_incident_report)
    assert ClaudeJudge is not None


@run_test
def test_supported_categories_match_registry():
    reg = default_registry()
    assert set(reg.keys()) == set(SUPPORTED_CATEGORIES), (
        f"registry={set(reg.keys())} supported={set(SUPPORTED_CATEGORIES)}"
    )
    for cat, ev in reg.items():
        assert isinstance(ev, MetricEvaluator), f"{cat} -> {type(ev)}"
        assert ev.metric_name, f"{cat} has empty metric_name"


@run_test
def test_sampling_policy_critical_always_runs():
    task = _make_task("inc-c", EvaluationCategory.RAG, Severity.CRITICAL)
    sampled, reason = _should_sample(task, DEFAULT_SAMPLING_POLICY)
    assert sampled is True, reason


@run_test
def test_sampling_policy_info_always_skipped():
    task = _make_task("inc-i", EvaluationCategory.RAG, Severity.INFO)
    sampled, reason = _should_sample(task, DEFAULT_SAMPLING_POLICY)
    assert sampled is False
    assert "rate=0" in reason or "priority=info" in reason


@run_test
def test_sampling_is_deterministic():
    task = _make_task("inc-stable", EvaluationCategory.RAG, Severity.RISK)
    a, _ = _should_sample(task, DEFAULT_SAMPLING_POLICY)
    b, _ = _should_sample(task, DEFAULT_SAMPLING_POLICY)
    assert a == b, "Sampling decisions must be deterministic"


@run_test
def test_sampling_rates_approximate_targets():
    # Over many synthetic ids, the sampled fraction should track the policy.
    rng_check = {Severity.CRITICAL: 0, Severity.RISK: 0, Severity.WARNING: 0}
    n = 1000
    for i in range(n):
        for sev in rng_check:
            t = _make_task(f"inc-{i}", EvaluationCategory.RAG, sev)
            sampled, _ = _should_sample(t, DEFAULT_SAMPLING_POLICY)
            if sampled:
                rng_check[sev] += 1
    assert rng_check[Severity.CRITICAL] == n
    risk_rate = rng_check[Severity.RISK] / n
    warn_rate = rng_check[Severity.WARNING] / n
    assert 0.4 < risk_rate < 0.6, f"risk_rate={risk_rate}"
    assert 0.05 < warn_rate < 0.15, f"warn_rate={warn_rate}"


@run_test
def test_case_builder_prompt():
    task = _make_task(
        "inc-p",
        EvaluationCategory.PROMPT,
        Severity.CRITICAL,
        context={
            "trace_input": "what's the weather?",
            "trace_output": "It is sunny.",
            "suspicious_payloads": ["ignore previous instructions"],
        },
    )
    case = build_test_case(task)
    assert "weather" in case.input
    assert "ignore previous instructions" in case.input
    assert case.actual_output == "It is sunny."


@run_test
def test_case_builder_rag_has_retrieval_context():
    task = _make_task(
        "inc-r",
        EvaluationCategory.RAG,
        Severity.RISK,
        context={
            "trace_input": "who painted the Mona Lisa?",
            "trace_output": "Leonardo da Vinci.",
            "retrieval_context": [
                {"span_id": "s1", "tool_name": "vector_search", "input": "q", "output": "Mona Lisa was painted by Leonardo."},
                {"span_id": "s2", "tool_name": "kb_lookup", "input": "q2", "output": "Leonardo lived in Italy."},
            ],
        },
    )
    case = build_test_case(task)
    assert isinstance(case.retrieval_context, list)
    assert len(case.retrieval_context) == 2
    assert any("Leonardo" in c for c in case.retrieval_context)


@run_test
def test_case_builder_tool_inlines_call_summary():
    task = _make_task(
        "inc-t",
        EvaluationCategory.TOOL_INVOCATION,
        Severity.WARNING,
        context={
            "primary_span_input": "find a flight",
            "primary_span_output": "booked",
            "tool_calls": [
                {"tool_name": "flight_search", "input": "JFK->SFO", "output": "ok"}
            ],
        },
    )
    case = build_test_case(task)
    assert "find a flight" in case.input
    assert "flight_search" in case.actual_output


@run_test
def test_case_builder_generation_minimal():
    task = _make_task(
        "inc-g",
        EvaluationCategory.LLM_GENERATION,
        Severity.RISK,
        context={
            "primary_span_input": "summarise X",
            "primary_span_output": "X is foo",
        },
    )
    case = build_test_case(task)
    assert case.input == "summarise X"
    assert case.actual_output == "X is foo"


async def _run_with_mocks(
    report: IncidentReport, raise_for: set[str] | None = None
) -> tuple[BatchEvaluationResult, MockEvaluator]:
    evaluator = MockEvaluator(raise_for=raise_for or set())
    registry = {cat: evaluator for cat in SUPPORTED_CATEGORIES}
    batch = await evaluate_incident_report_async(
        report,
        judge=MockJudge(),  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
    )
    return batch, evaluator


@run_test
async def test_runner_skips_info_and_unsupported_categories():
    report = _make_report(
        _make_task("inc-1", EvaluationCategory.RAG, Severity.CRITICAL),
        _make_task("inc-2", EvaluationCategory.LLM_GENERATION, Severity.INFO),
        _make_task("inc-3", EvaluationCategory.OBSERVABILITY, Severity.CRITICAL),
    )
    batch, ev = await _run_with_mocks(report)
    assert batch.n_tasks_total == 3
    assert batch.n_tasks_sampled == 1
    assert batch.n_tasks_skipped == 2
    assert [t.incident_id for t in ev.calls] == ["inc-1"]


@run_test
async def test_runner_groups_outcomes_by_incident():
    report = _make_report(
        _make_task("inc-A", EvaluationCategory.RAG, Severity.CRITICAL),
        _make_task("inc-A", EvaluationCategory.LLM_GENERATION, Severity.CRITICAL),
        _make_task("inc-B", EvaluationCategory.PROMPT, Severity.CRITICAL),
    )
    batch, _ = await _run_with_mocks(report)
    by_id = {e.incident_id: e for e in batch.evaluations}
    assert "inc-A" in by_id and "inc-B" in by_id
    assert len(by_id["inc-A"].outcomes) == 2
    assert len(by_id["inc-B"].outcomes) == 1


@run_test
async def test_runner_failure_isolation():
    report = _make_report(
        _make_task("inc-good", EvaluationCategory.RAG, Severity.CRITICAL),
        _make_task("inc-bad", EvaluationCategory.LLM_GENERATION, Severity.CRITICAL),
    )
    batch, _ = await _run_with_mocks(report, raise_for={"inc-bad"})
    assert batch.n_tasks_sampled == 2
    assert batch.n_tasks_errored == 1
    bad = [o for o in batch.outcomes() if o.incident_id == "inc-bad"]
    good = [o for o in batch.outcomes() if o.incident_id == "inc-good"]
    assert bad[0].error is not None
    assert good[0].passed is True


@run_test
def test_serialisation_round_trip():
    report = _make_report(
        _make_task("inc-S", EvaluationCategory.RAG, Severity.CRITICAL),
    )
    batch, _ = asyncio.run(_run_with_mocks(report))
    payload = batch.as_dict()
    assert payload["trace_id"] == "trace-xyz"
    assert payload["n_tasks_sampled"] == 1
    assert "evaluations" in payload
    assert payload["sampling_policy"]["critical"] == 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"deep_evaluation smoke tests: {len(PASSED) + len(FAILED)} executed")
    print(f"  passed: {len(PASSED)}")
    print(f"  failed: {len(FAILED)}")
    for name, msg in FAILED:
        print(f"\n--- {name} ---\n{msg}")
    sys.exit(0 if not FAILED else 1)
