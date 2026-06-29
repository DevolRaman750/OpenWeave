"""
incident_classification — Enrich → correlate → classify normalized flags.

Pipeline-of-pipelines
---------------------
    trace_id
      → anomaly_pipeline.run_anomaly_pipeline_full(...)
      → (TraceAnomalyReport, TraceBundle)
      → classify_trace(report, bundle)
      → IncidentReport
            ├── incidents    : correlated EnrichedFlag groups
            ├── categories   : multi-label EvaluationCategory tags
            └── evaluation_plan : ready-to-execute EvaluationTask list for
                                  the next stage (DeepEval, etc.)

One-shot convenience
--------------------
    from openweave_core.incident_classification import classify_trace_by_id

    report = classify_trace_by_id("<trace_id>")
    for incident in report.incidents:
        print(incident.severity, incident.categories, incident.primary_subject_id)
"""

from openweave_core.incident_classification.api import (
    classify_trace,
    classify_trace_async,
    classify_trace_by_id,
    classify_trace_by_id_async,
    classify_trace_for_spans,
)
from openweave_core.incident_classification.classification import (
    CategoryClassifier,
    RuleBasedClassifier,
    classify_incidents,
)
from openweave_core.incident_classification.contracts import (
    EnrichedFlag,
    EvaluationCategory,
    EvaluationTask,
    Incident,
    IncidentReport,
)
from openweave_core.incident_classification.correlation import (
    IncidentCorrelator,
    correlate_flags,
)
from openweave_core.incident_classification.enrichment import FlagEnricher

__all__ = [
    # Entry points
    "classify_trace",
    "classify_trace_async",
    "classify_trace_by_id",
    "classify_trace_by_id_async",
    "classify_trace_for_spans",
    # Output types
    "IncidentReport",
    "Incident",
    "EnrichedFlag",
    "EvaluationTask",
    "EvaluationCategory",
    # Implementations
    "FlagEnricher",
    "IncidentCorrelator",
    "correlate_flags",
    "RuleBasedClassifier",
    "CategoryClassifier",
    "classify_incidents",
]
