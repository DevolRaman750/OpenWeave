"""
contracts.py — Shared dataclasses for the unified anomaly pipeline.

Defines:
  * TraceBundle          : everything a detector needs as input (parsed once).
  * NormalizedFlag       : the *uniform* anomaly representation produced by
                           every detector adapter; downstream classification
                           layers (RAG / Prompt / LLM / Tool) consume this.
  * DetectorResult       : per-detector envelope (timing, errors, raw summary).
  * TraceAnomalyReport   : final combined output of one pipeline run.

Severity is re-exported from ``sentinel_agent.findings.Severity`` so the
whole platform speaks one severity ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from openweave_core.models.span import ParsedSpan
from openweave_core.models.trace_envelope import TraceEnvelope
from openweave_core.sentinel_agent.findings import Severity


# ---------------------------------------------------------------------------
# Canonical pipeline + subject labels
# ---------------------------------------------------------------------------

# Source pipelines — kept as plain strings (not Enum) so external classifiers
# can extend with new detector names without code changes.
SOURCE_ADAPTIVE_BASELINE: str = "adaptive_baseline"
SOURCE_SENTINEL_AGENT:    str = "sentinel_agent"
SOURCE_CYCLE_DETECTION:   str = "cycle_detection"

ALL_SOURCES: tuple[str, ...] = (
    SOURCE_ADAPTIVE_BASELINE,
    SOURCE_SENTINEL_AGENT,
    SOURCE_CYCLE_DETECTION,
)

# Subject types — what kind of entity a flag refers to.
SUBJECT_SPAN:        str = "span"
SUBJECT_NODE:        str = "node"
SUBJECT_EDGE:        str = "edge"
SUBJECT_ATTACK_PATH: str = "attack_path"
SUBJECT_SPAN_PAIR:   str = "span_pair"
SUBJECT_TRACE:       str = "trace"


# ---------------------------------------------------------------------------
# TraceBundle — the shared input fan-out
# ---------------------------------------------------------------------------

@dataclass
class TraceBundle:
    """Parsed trace + pre-built indexes + trace-level envelope.

    Built once by the pipeline orchestrator after fetching/parsing; passed to
    every detector adapter (and to the downstream classifier) so nobody
    re-parses or re-indexes redundantly.

    Attributes
    ----------
    trace_id        : the trace identifier.
    spans           : flat ParsedSpan list as returned by the parser.
    span_by_id      : span_id → ParsedSpan index for O(1) lookup.
    parent_index    : span_id → parent span_id (omitted for roots).
    children_index  : span_id → list of child span_ids.
    root_span_ids   : span_ids with no parent (entry points to the trace).
    envelope        : trace-level context (input/output/session/tags/metadata).
                      Always present; may be ``TraceEnvelope.empty(trace_id)``
                      when the caller supplied spans directly.
    extras          : free-form per-run cache (e.g. memoised dynamic graph) —
                      detectors / classifiers may write into this to share
                      derived data between fan-out branches when needed.
    """

    trace_id: str
    spans: list[ParsedSpan]
    span_by_id: dict[str, ParsedSpan] = field(default_factory=dict)
    parent_index: dict[str, str] = field(default_factory=dict)
    children_index: dict[str, list[str]] = field(default_factory=dict)
    root_span_ids: list[str] = field(default_factory=list)
    envelope: TraceEnvelope = field(
        default_factory=lambda: TraceEnvelope.empty("")
    )
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spans(
        cls,
        trace_id: str,
        spans: list[ParsedSpan],
        *,
        envelope: Optional[TraceEnvelope] = None,
    ) -> "TraceBundle":
        spans = list(spans)
        span_by_id = {s.id: s for s in spans}
        parent_index: dict[str, str] = {}
        children_index: dict[str, list[str]] = {}
        root_span_ids: list[str] = []
        for s in spans:
            if s.parent_id:
                parent_index[s.id] = s.parent_id
                children_index.setdefault(s.parent_id, []).append(s.id)
            else:
                root_span_ids.append(s.id)
        return cls(
            trace_id=trace_id,
            spans=spans,
            span_by_id=span_by_id,
            parent_index=parent_index,
            children_index=children_index,
            root_span_ids=root_span_ids,
            envelope=envelope or TraceEnvelope.empty(trace_id),
        )

    # ----- query helpers --------------------------------------------------

    def parent_of(self, span_id: str) -> Optional[ParsedSpan]:
        pid = self.parent_index.get(span_id)
        return self.span_by_id.get(pid) if pid else None

    def children_of(self, span_id: str) -> list[ParsedSpan]:
        return [
            self.span_by_id[cid]
            for cid in self.children_index.get(span_id, ())
            if cid in self.span_by_id
        ]

    def siblings_of(self, span_id: str) -> list[ParsedSpan]:
        pid = self.parent_index.get(span_id)
        if pid is None:
            return []
        return [
            self.span_by_id[cid]
            for cid in self.children_index.get(pid, ())
            if cid != span_id and cid in self.span_by_id
        ]

    def root_of(self, span_id: str) -> Optional[ParsedSpan]:
        """Walk parents up to the trace root for *span_id*. Cycle-safe."""
        seen: set[str] = set()
        current = span_id
        while current in self.parent_index and current not in seen:
            seen.add(current)
            current = self.parent_index[current]
        return self.span_by_id.get(current)


# ---------------------------------------------------------------------------
# NormalizedFlag — the platform-wide anomaly contract
# ---------------------------------------------------------------------------

@dataclass
class NormalizedFlag:
    """One anomaly, in a detector-agnostic shape.

    Every detector adapter emits these. The final report aggregates them, and
    downstream classifiers (RAG / Prompt / LLM / Tool) group + label them
    without needing to know which engine produced them.

    Attributes
    ----------
    id              : stable identifier unique within a report.
    source_pipeline : SOURCE_* constant — which detector produced this flag.
    category        : detector-specific category code, e.g. ``"joint_anomaly"``,
                      ``"EDGE_UNAUTHORIZED_TOOL"``, ``"redundant_cycle"``.
                      Free-form so future classifier layers can route on it.
    severity        : INFO < WARNING < RISK < CRITICAL.
    subject_id      : id of the entity the flag concerns (span / node / edge / …).
    subject_type    : SUBJECT_* constant — disambiguates id namespaces.
    confidence      : detector confidence in [0.0, 1.0].
    message         : human-readable summary.
    evidence        : detector-specific structured payload — full audit trail
                      (raw scores, thresholds, matched patterns, etc.).
    timestamp       : when in the trace the offending event occurred
                      (None for whole-trace findings such as attack paths).
    detected_at     : when the pipeline produced this flag (UTC).
    metadata        : free-form extras.
    """

    id: str
    source_pipeline: str
    category: str
    severity: Severity
    subject_id: str
    subject_type: str
    confidence: float = 1.0
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_pipeline": self.source_pipeline,
            "category": self.category,
            "severity": self.severity.value,
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "confidence": self.confidence,
            "message": self.message,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# DetectorResult — per-engine envelope (with errors + timing)
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """Outcome of one detector's run.

    Successful runs carry ``flags``; failed runs carry ``error`` and the
    pipeline continues with the remaining detectors.
    ``warning`` is for non-fatal degradation (e.g. embedding model
    unavailable — we still return structural cycle flags, just no semantic
    confirmation).
    """

    detector: str
    ok: bool
    flags: list[NormalizedFlag] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    warning: Optional[str] = None
    raw_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def flag_count(self) -> int:
        return len(self.flags)

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "ok": self.ok,
            "flag_count": self.flag_count,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "warning": self.warning,
            "raw_summary": self.raw_summary,
        }


# ---------------------------------------------------------------------------
# TraceAnomalyReport — pipeline's final output
# ---------------------------------------------------------------------------

@dataclass
class TraceAnomalyReport:
    """End-to-end result of one ``run_anomaly_pipeline`` call.

    Attributes
    ----------
    trace_id                 : the analysed trace's id.
    flagged                  : True iff at least one detector emitted a flag.
    flags                    : merged, source-attributed flag list — what the
                               classification layer consumes.
    by_source                : {source_pipeline → flag_count}.
    detector_results         : the per-detector envelopes with timing/errors.
    overall_severity         : max severity across all flags.
    parse_duration_seconds   : time spent fetching + parsing (single, shared).
    total_duration_seconds   : wall-clock duration of the pipeline run.
    generated_at             : when the report was produced.
    metadata                 : free-form extras (n_spans, system_spec_source…).
    """

    trace_id: str
    flagged: bool
    flags: list[NormalizedFlag] = field(default_factory=list)
    by_source: dict[str, int] = field(default_factory=dict)
    detector_results: list[DetectorResult] = field(default_factory=list)
    overall_severity: Severity = Severity.INFO
    parse_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- queries ---------------------------------------------------------

    def by_severity(self, severity: Severity) -> list[NormalizedFlag]:
        return [f for f in self.flags if f.severity == severity]

    def by_source_pipeline(self, source: str) -> list[NormalizedFlag]:
        return [f for f in self.flags if f.source_pipeline == source]

    def by_subject(self, subject_id: str) -> list[NormalizedFlag]:
        return [f for f in self.flags if f.subject_id == subject_id]

    def failed_detectors(self) -> list[str]:
        return [r.detector for r in self.detector_results if not r.ok]

    def has_anomaly(self) -> bool:
        return any(
            f.severity in (Severity.RISK, Severity.CRITICAL) for f in self.flags
        )

    def has_critical(self) -> bool:
        return any(f.severity == Severity.CRITICAL for f in self.flags)

    # -- serialisation ---------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "flagged": self.flagged,
            "overall_severity": self.overall_severity.value,
            "by_source": dict(self.by_source),
            "failed_detectors": self.failed_detectors(),
            "parse_duration_seconds": self.parse_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "generated_at": self.generated_at.isoformat(),
            "flags": [f.as_dict() for f in self.flags],
            "detector_results": [r.as_dict() for r in self.detector_results],
            "metadata": self.metadata,
        }
