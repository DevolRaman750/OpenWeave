"""
deep_evaluation — DeepEval wrapper for the 4 OpenWeave eval categories.

Pipeline
--------
    IncidentReport
      → evaluate_incident_report(report)
      → BatchEvaluationResult
            ├── evaluations  : list[IncidentEvaluation]
            └── outcomes()   : flat list[EvaluationOutcome]

One-shot usage
--------------
    from openweave_core.deep_evaluation import evaluate_incident_report

    batch = evaluate_incident_report(incident_report)
    for outcome in batch.outcomes():
        print(outcome.category, outcome.metric_name, outcome.score, outcome.passed)

Categories supported (1 metric each)
------------------------------------
    PROMPT          : GEval custom criterion (prompt-injection resistance)
    RAG             : FaithfulnessMetric
    TOOL_INVOCATION : GEval custom criterion (tool correctness)
    LLM_GENERATION  : AnswerRelevancyMetric

OBSERVABILITY and SAFETY tasks are silently skipped — those signals are
already covered by the upstream anomaly pipeline + sentinel.

Sampling policy
---------------
    CRITICAL: 1.0  RISK: 0.5  WARNING: 0.1  INFO: 0.0

Configuration (env vars)
------------------------
    OPENWEAVE_JUDGE_MODEL        — default ``claude-sonnet-4-6``
    OPENWEAVE_JUDGE_MAX_TOKENS   — default 2048
    OPENWEAVE_JUDGE_TEMP         — default 0.0
    OPENWEAVE_JUDGE_CONCURRENCY  — default 8
    ANTHROPIC_API_KEY            — required for live calls
"""

from openweave_core.deep_evaluation.contracts import (
    BatchEvaluationResult,
    EvaluationCategory,
    EvaluationOutcome,
    EvaluationTask,
    IncidentEvaluation,
)
from openweave_core.deep_evaluation.judge import ClaudeJudge, DEFAULT_JUDGE_MODEL
from openweave_core.deep_evaluation.metric_registry import (
    AnswerRelevancyEvaluator,
    FaithfulnessEvaluator,
    MetricEvaluator,
    PromptInjectionEvaluator,
    SUPPORTED_CATEGORIES,
    ToolCorrectnessEvaluator,
    default_registry,
)
from openweave_core.deep_evaluation.runner import (
    DEFAULT_SAMPLING_POLICY,
    evaluate_incident_report,
    evaluate_incident_report_async,
)

__all__ = [
    # Entry points
    "evaluate_incident_report",
    "evaluate_incident_report_async",
    # Output types
    "BatchEvaluationResult",
    "IncidentEvaluation",
    "EvaluationOutcome",
    "EvaluationTask",
    "EvaluationCategory",
    # Judge
    "ClaudeJudge",
    "DEFAULT_JUDGE_MODEL",
    # Registry
    "MetricEvaluator",
    "default_registry",
    "SUPPORTED_CATEGORIES",
    "AnswerRelevancyEvaluator",
    "FaithfulnessEvaluator",
    "PromptInjectionEvaluator",
    "ToolCorrectnessEvaluator",
    # Policy
    "DEFAULT_SAMPLING_POLICY",
]
