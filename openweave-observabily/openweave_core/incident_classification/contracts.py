"""
contracts.py — Types for the incident enrichment + correlation + classification layer.

Three conceptual levels, three concrete types:

    Flag     ────►  EnrichedFlag    : NormalizedFlag + resolved trace context.
    Group    ────►  Incident        : correlated EnrichedFlags + categories.
    Trace    ────►  IncidentReport  : final per-trace classification summary.

A fourth output, ``EvaluationTask``, is a ready-to-execute pointer that the
next stage (DeepEval, custom evaluator, …) can iterate on without doing more
analysis itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from openweave_core.anomaly_pipeline.contracts import NormalizedFlag
from openweave_core.models.span import ParsedSpan
from openweave_core.models.trace_envelope import TraceEnvelope
from openweave_core.sentinel_agent.findings import Severity
from openweave_core.sentinel_agent.schema import Edge, Node


# ---------------------------------------------------------------------------
# Evaluation categories — the multi-label routing target
# ---------------------------------------------------------------------------

class EvaluationCategory(str, Enum):
    """Buckets used by the downstream evaluator (DeepEval, etc.) to choose
    which metrics / checks to run for each incident."""

    PROMPT = "prompt"                  # injection, jailbreak, instruction override
    TOOL_INVOCATION = "tool_invocation"
    LLM_GENERATION = "llm_generation"  # reasoning, hallucination, output quality
    RAG = "rag"                        # retrieval, grounding, context loops
    OBSERVABILITY = "observability"    # cost / latency / axis or joint drift
    SAFETY = "safety"                  # secret leak / contract violation umbrella


# ---------------------------------------------------------------------------
# Enriched flag — NormalizedFlag + resolved trace/graph context
# ---------------------------------------------------------------------------

@dataclass
class EnrichedFlag:
    """A NormalizedFlag with resolved subject + surrounding context attached.

    Why
    ---
    Classification needs more than the flag itself: which actual span fired?
    Was it a generation span, a tool call, or a retrieval lookup? What is
    the parent / sibling context? The enricher resolves these from the
    TraceBundle (and sentinel's dynamic graph, when present) so the
    correlator and classifier do not re-do that work.

    Attributes
    ----------
    flag             : original NormalizedFlag (untouched).
    span             : ParsedSpan the flag targets (if subject is a span /
                       span pair / edge owning a span).
    paired_span      : second span when the subject is a SPAN_PAIR.
    parent_span      : span_by_id[span.parent_id], if any.
    children_spans   : the span's children, in order.
    sibling_spans    : same-parent peers.
    root_span        : the trace's root ancestor of ``span``.
    node             : sentinel AgentNode/ToolNode when subject is a NODE.
    edge             : sentinel Edge when subject is an EDGE.
    affected_span_ids: every span_id this flag implicates (for correlation).
    tool_name        : best-effort tool name (only set for tool spans / nodes).
    is_tool          : True iff the flag concerns a tool span / tool node /
                       tool-invocation edge.
    is_generation    : True iff the underlying span is a GENERATION.
    is_retrieval     : True iff the tool name looks like a retrieval endpoint.
    payload_texts    : aggregated free-text seen in this context — used by
                       the classifier's keyword rules.
    extras           : free-form attachments.
    """

    flag: NormalizedFlag

    # Resolved subjects
    span: Optional[ParsedSpan] = None
    paired_span: Optional[ParsedSpan] = None
    parent_span: Optional[ParsedSpan] = None
    children_spans: list[ParsedSpan] = field(default_factory=list)
    sibling_spans: list[ParsedSpan] = field(default_factory=list)
    root_span: Optional[ParsedSpan] = None
    node: Optional[Node] = None
    edge: Optional[Edge] = None

    # Derived
    affected_span_ids: set[str] = field(default_factory=set)
    tool_name: Optional[str] = None
    is_tool: bool = False
    is_generation: bool = False
    is_retrieval: bool = False

    # Aggregated text used by the rule classifier
    payload_texts: list[str] = field(default_factory=list)

    extras: dict[str, Any] = field(default_factory=dict)

    # -- pass-through accessors -------------------------------------------

    @property
    def id(self) -> str:
        return self.flag.id

    @property
    def severity(self) -> Severity:
        return self.flag.severity

    @property
    def source_pipeline(self) -> str:
        return self.flag.source_pipeline

    @property
    def category(self) -> str:
        return self.flag.category

    @property
    def confidence(self) -> float:
        return self.flag.confidence

    @property
    def timestamp(self) -> Optional[datetime]:
        return self.flag.timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag.as_dict(),
            "span_id": self.span.id if self.span else None,
            "paired_span_id": self.paired_span.id if self.paired_span else None,
            "parent_span_id": self.parent_span.id if self.parent_span else None,
            "root_span_id": self.root_span.id if self.root_span else None,
            "node_id": self.node.id if self.node else None,
            "edge_id": self.edge.id if self.edge else None,
            "tool_name": self.tool_name,
            "is_tool": self.is_tool,
            "is_generation": self.is_generation,
            "is_retrieval": self.is_retrieval,
            "affected_span_ids": sorted(self.affected_span_ids),
        }


# ---------------------------------------------------------------------------
# Incident — correlated EnrichedFlags
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    """A correlated group of EnrichedFlag's representing one issue episode.

    Attributes
    ----------
    id                 : stable id within an IncidentReport (e.g. ``"inc-1"``).
    flags              : the enriched flags that belong to this incident.
    title              : short, human-readable headline.
    description        : longer summary explaining what's happening.
    primary_subject_id : the most-implicated subject id.
    subject_ids        : every subject id touched by member flags.
    affected_span_ids  : union of EnrichedFlag.affected_span_ids.
    source_pipelines   : which detectors contributed.
    severity           : max severity across member flags.
    confidence         : confidence-weighted, severity-tilted aggregate.
    started_at, ended_at : earliest / latest timestamps among member flags.
    categories         : multi-label classification output.
    category_evidence  : per-category list of rule names that fired
                         (audit trail for the classifier).
    correlation_signals: rule names that joined the member flags together
                         (audit trail for the correlator).
    """

    id: str
    flags: list[EnrichedFlag]
    title: str = ""
    description: str = ""
    primary_subject_id: str = ""
    subject_ids: set[str] = field(default_factory=set)
    affected_span_ids: set[str] = field(default_factory=set)
    source_pipelines: set[str] = field(default_factory=set)
    severity: Severity = Severity.INFO
    confidence: float = 0.0
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    categories: list[EvaluationCategory] = field(default_factory=list)
    category_evidence: dict[EvaluationCategory, list[str]] = field(
        default_factory=dict
    )
    correlation_signals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "primary_subject_id": self.primary_subject_id,
            "subject_ids": sorted(self.subject_ids),
            "affected_span_ids": sorted(self.affected_span_ids),
            "source_pipelines": sorted(self.source_pipelines),
            "severity": self.severity.value,
            "confidence": self.confidence,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "categories": [c.value for c in self.categories],
            "category_evidence": {
                k.value: list(v) for k, v in self.category_evidence.items()
            },
            "correlation_signals": list(self.correlation_signals),
            "flag_ids": [f.id for f in self.flags],
            "flag_count": len(self.flags),
        }


# ---------------------------------------------------------------------------
# Evaluation task — pointer to a downstream evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvaluationTask:
    """One concrete evaluation the downstream framework should run.

    Attributes
    ----------
    incident_id        : the Incident this task targets.
    category           : the EvaluationCategory bucket.
    priority           : copy of the incident's severity (CRITICAL first).
    subject_id         : the primary subject the evaluator should focus on.
    context            : structured payload the evaluator can consume directly
                         (input/output texts, tool args, retrieval context …).
    suggested_metrics  : free-form list of metric names — e.g.
                         ``["deepeval.metrics.HallucinationMetric"]`` —
                         informational only; the downstream layer maps them
                         to its actual metric registry.
    """

    incident_id: str
    category: EvaluationCategory
    priority: Severity
    subject_id: str
    context: dict[str, Any] = field(default_factory=dict)
    suggested_metrics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "category": self.category.value,
            "priority": self.priority.value,
            "subject_id": self.subject_id,
            "context": self.context,
            "suggested_metrics": list(self.suggested_metrics),
        }


# ---------------------------------------------------------------------------
# Incident report — top-level deliverable
# ---------------------------------------------------------------------------

@dataclass
class IncidentReport:
    """Final classification output for one trace."""

    trace_id: str
    envelope: Optional[TraceEnvelope] = None
    incidents: list[Incident] = field(default_factory=list)
    category_summary: dict[EvaluationCategory, int] = field(default_factory=dict)
    source_summary: dict[str, int] = field(default_factory=dict)
    overall_severity: Severity = Severity.INFO
    evaluation_plan: list[EvaluationTask] = field(default_factory=list)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- queries ----------------------------------------------------------

    def by_category(self, category: EvaluationCategory) -> list[Incident]:
        return [i for i in self.incidents if category in i.categories]

    def by_severity(self, severity: Severity) -> list[Incident]:
        return [i for i in self.incidents if i.severity == severity]

    def has_anomaly(self) -> bool:
        return any(
            i.severity in (Severity.RISK, Severity.CRITICAL) for i in self.incidents
        )

    def has_critical(self) -> bool:
        return any(i.severity == Severity.CRITICAL for i in self.incidents)

    # -- serialisation ---------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "overall_severity": self.overall_severity.value,
            "category_summary": {
                k.value: v for k, v in self.category_summary.items()
            },
            "source_summary": dict(self.source_summary),
            "generated_at": self.generated_at.isoformat(),
            "envelope": self.envelope.as_dict() if self.envelope else None,
            "incidents": [i.as_dict() for i in self.incidents],
            "evaluation_plan": [t.as_dict() for t in self.evaluation_plan],
            "metadata": self.metadata,
        }
