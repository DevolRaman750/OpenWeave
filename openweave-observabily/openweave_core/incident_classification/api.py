"""
api.py — One-call entry points for the incident-classification stage.

    classify_trace(report, bundle)        : sync, no fetching, uses upstream pipeline output.
    classify_trace_by_id(trace_id, ...)    : sync, runs the unified anomaly pipeline first.
    classify_trace_for_spans(spans, ...)   : sync, skips the parser entirely.
    classify_trace_async(...)              : async variants for FastAPI / Jupyter.

All three return an ``IncidentReport`` with:
    * enriched flags grouped into incidents,
    * multi-label EvaluationCategory tags per incident,
    * an ``evaluation_plan`` of EvaluationTask pointers ready for the next
      stage (DeepEval or similar) to iterate on.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from openweave_core.anomaly_pipeline.contracts import (
    NormalizedFlag,
    TraceAnomalyReport,
    TraceBundle,
)
from openweave_core.anomaly_pipeline.pipeline import (
    FullParser,
    Parser,
    run_anomaly_pipeline_for_spans_full_async,
    run_anomaly_pipeline_full_async,
)
from openweave_core.models.span import ParsedSpan
from openweave_core.models.trace_envelope import TraceEnvelope
from openweave_core.sentinel_agent import SystemSpec
from openweave_core.sentinel_agent.findings import Severity

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
from openweave_core.incident_classification.correlation import IncidentCorrelator
from openweave_core.incident_classification.enrichment import FlagEnricher


_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.RISK: 2,
    Severity.CRITICAL: 3,
}


# ---------------------------------------------------------------------------
# Suggested metric registry — informational hint for the next stage
# ---------------------------------------------------------------------------

_SUGGESTED_METRICS: dict[EvaluationCategory, tuple[str, ...]] = {
    EvaluationCategory.PROMPT: (
        "deepeval.metrics.PromptInjectionMetric",
        "deepeval.metrics.JailbreakDetectionMetric",
    ),
    EvaluationCategory.TOOL_INVOCATION: (
        "deepeval.metrics.ToolCorrectnessMetric",
        "openweave.metrics.UnauthorizedToolUseMetric",
    ),
    EvaluationCategory.LLM_GENERATION: (
        "deepeval.metrics.HallucinationMetric",
        "deepeval.metrics.AnswerRelevancyMetric",
        "deepeval.metrics.BiasMetric",
    ),
    EvaluationCategory.RAG: (
        "deepeval.metrics.ContextualRelevancyMetric",
        "deepeval.metrics.ContextualRecallMetric",
        "deepeval.metrics.FaithfulnessMetric",
    ),
    EvaluationCategory.OBSERVABILITY: (
        "openweave.metrics.CostBudgetMetric",
        "openweave.metrics.LatencyBudgetMetric",
    ),
    EvaluationCategory.SAFETY: (
        "deepeval.metrics.ToxicityMetric",
        "openweave.metrics.SecretLeakMetric",
    ),
}


# ---------------------------------------------------------------------------
# Core sync entry — the layer the user is documenting
# ---------------------------------------------------------------------------

def classify_trace(
    report: TraceAnomalyReport,
    bundle: TraceBundle,
    *,
    classifier: Optional[CategoryClassifier] = None,
) -> IncidentReport:
    """Take a TraceAnomalyReport + its TraceBundle; return an IncidentReport.

    Steps
    -----
    1. Enrich each NormalizedFlag with span / graph / envelope context.
    2. Correlate enriched flags into Incidents via the union-find correlator.
    3. Classify each Incident into one or more EvaluationCategory buckets.
    4. Build the evaluation plan (one EvaluationTask per (incident, category)).
    """
    enricher = FlagEnricher(bundle)
    enriched: list[EnrichedFlag] = enricher.enrich_all(report)

    correlator = IncidentCorrelator()
    incidents = correlator.correlate(enriched)

    classify_incidents(incidents, classifier=classifier)

    return _assemble_report(report, bundle, incidents)


def classify_trace_by_id(
    trace_id: str,
    *,
    system_spec: Optional[SystemSpec] = None,
    parser: Optional[Parser] = None,
    full_parser: Optional[FullParser] = None,
    classifier: Optional[CategoryClassifier] = None,
) -> IncidentReport:
    """One-shot: parse + run anomaly pipeline + classify, all in one call."""
    report, bundle = asyncio.run(
        run_anomaly_pipeline_full_async(
            trace_id,
            system_spec=system_spec,
            parser=parser,
            full_parser=full_parser,
        )
    )
    return classify_trace(report, bundle, classifier=classifier)


def classify_trace_for_spans(
    spans: list[ParsedSpan],
    *,
    trace_id: str,
    envelope: Optional[TraceEnvelope] = None,
    system_spec: Optional[SystemSpec] = None,
    classifier: Optional[CategoryClassifier] = None,
) -> IncidentReport:
    """Skip the parser — caller already has the spans."""
    report, bundle = asyncio.run(
        run_anomaly_pipeline_for_spans_full_async(
            spans,
            trace_id=trace_id,
            envelope=envelope,
            system_spec=system_spec,
        )
    )
    return classify_trace(report, bundle, classifier=classifier)


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------

async def classify_trace_async(
    report: TraceAnomalyReport,
    bundle: TraceBundle,
    *,
    classifier: Optional[CategoryClassifier] = None,
) -> IncidentReport:
    """Async-compatible classify; the work itself is synchronous & light."""
    return await asyncio.to_thread(
        classify_trace, report, bundle, classifier=classifier
    )


async def classify_trace_by_id_async(
    trace_id: str,
    *,
    system_spec: Optional[SystemSpec] = None,
    parser: Optional[Parser] = None,
    full_parser: Optional[FullParser] = None,
    classifier: Optional[CategoryClassifier] = None,
) -> IncidentReport:
    report, bundle = await run_anomaly_pipeline_full_async(
        trace_id,
        system_spec=system_spec,
        parser=parser,
        full_parser=full_parser,
    )
    return await classify_trace_async(report, bundle, classifier=classifier)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _assemble_report(
    upstream: TraceAnomalyReport,
    bundle: TraceBundle,
    incidents: list[Incident],
) -> IncidentReport:
    # Aggregate categories + sources across all incidents.
    category_summary: dict[EvaluationCategory, int] = {}
    source_summary: dict[str, int] = dict(upstream.by_source)
    for inc in incidents:
        for c in inc.categories:
            category_summary[c] = category_summary.get(c, 0) + 1

    overall_severity = max(
        (inc.severity for inc in incidents),
        key=lambda s: _SEVERITY_RANK[s],
        default=Severity.INFO,
    )

    plan = _build_evaluation_plan(bundle, incidents)

    return IncidentReport(
        trace_id=upstream.trace_id,
        envelope=bundle.envelope,
        incidents=incidents,
        category_summary=category_summary,
        source_summary=source_summary,
        overall_severity=overall_severity,
        evaluation_plan=plan,
        metadata={
            "n_incidents": len(incidents),
            "n_flags_in_input": len(upstream.flags),
            "n_spans": len(bundle.spans),
            "envelope_input_present": bool(bundle.envelope.input_text),
            "envelope_output_present": bool(bundle.envelope.output_text),
            "envelope_tags": list(bundle.envelope.tags),
        },
    )


def _build_evaluation_plan(
    bundle: TraceBundle, incidents: list[Incident]
) -> list[EvaluationTask]:
    plan: list[EvaluationTask] = []
    for inc in incidents:
        for category in inc.categories:
            plan.append(
                EvaluationTask(
                    incident_id=inc.id,
                    category=category,
                    priority=inc.severity,
                    subject_id=inc.primary_subject_id,
                    context=_build_task_context(bundle, inc, category),
                    suggested_metrics=list(_SUGGESTED_METRICS.get(category, ())),
                )
            )
    # Highest-priority tasks first; stable on incident id then category.
    plan.sort(
        key=lambda t: (
            -_SEVERITY_RANK[t.priority],
            t.incident_id,
            t.category.value,
        )
    )
    return plan


def _build_task_context(
    bundle: TraceBundle, incident: Incident, category: EvaluationCategory
) -> dict:
    """Build the context bag the downstream evaluator can consume directly."""
    affected_spans = [
        bundle.span_by_id[sid].id
        for sid in incident.affected_span_ids
        if sid in bundle.span_by_id
    ]

    primary_span = bundle.span_by_id.get(incident.primary_subject_id)
    primary_input = primary_span.input_text if primary_span else ""
    primary_output = primary_span.output_text if primary_span else ""

    # Trace-level envelope so RAG / Prompt classifiers can compare end-to-end
    # input vs. final output.
    env = bundle.envelope

    # Per-category surfaces — the evaluator looks here first.
    payload: dict = {
        "trace_input": env.input_text,
        "trace_output": env.output_text,
        "session_id": env.session_id,
        "user_id": env.user_id,
        "tags": list(env.tags),
        "environment": env.environment,
        "primary_span_id": incident.primary_subject_id,
        "primary_span_input": primary_input,
        "primary_span_output": primary_output,
        "affected_span_ids": affected_spans,
        "incident_title": incident.title,
        "category_evidence": [
            r for r in incident.category_evidence.get(category, [])
        ],
        "source_pipelines": sorted(incident.source_pipelines),
    }

    # RAG-specific: include the retrieval span text(s) explicitly.
    if category == EvaluationCategory.RAG:
        retrieval_spans = [
            bundle.span_by_id[sid]
            for sid in incident.affected_span_ids
            if sid in bundle.span_by_id
        ]
        payload["retrieval_context"] = [
            {
                "span_id": s.id,
                "tool_name": s.tool_name,
                "input": s.input_text,
                "output": s.output_text,
            }
            for s in retrieval_spans
            if s.tool_name
        ]

    # TOOL_INVOCATION-specific: include every tool call's args + output.
    if category == EvaluationCategory.TOOL_INVOCATION:
        tool_calls = []
        for ef in incident.flags:
            if ef.is_tool and ef.span is not None:
                tool_calls.append({
                    "span_id": ef.span.id,
                    "tool_name": ef.tool_name,
                    "input": ef.span.input_text,
                    "output": ef.span.output_text,
                    "evidence_flags": ef.flag.evidence.get("flags") or [],
                })
        payload["tool_calls"] = tool_calls

    # PROMPT-specific: aggregate every text payload the classifier flagged.
    if category == EvaluationCategory.PROMPT:
        payload["suspicious_payloads"] = [
            t for ef in incident.flags for t in ef.payload_texts
        ][:10]

    return payload
