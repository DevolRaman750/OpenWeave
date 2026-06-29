"""
findings.py — Output types for the Phase-2 Behavior Analyzer.

A Finding is one unit of evidence about anomalous behaviour — produced by
rule engines, LLM judges, or the attack-path matcher. A RiskReport bundles
all findings for a trace into a single explainable result with root-cause
attribution and tailored remediations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    """Ordered: INFO < WARNING < RISK < CRITICAL."""
    INFO = "info"
    WARNING = "warning"
    RISK = "risk"
    CRITICAL = "critical"

    @classmethod
    def max(cls, *severities: "Severity") -> "Severity":
        if not severities:
            return cls.INFO
        order = ("info", "warning", "risk", "critical")
        return cls(max(severities, key=lambda s: order.index(s.value)).value)


class FindingCategory(str, Enum):
    NODE = "node"
    EDGE = "edge"
    ATTACK_PATH = "attack_path"


class FindingSource(str, Enum):
    RULE_ENGINE = "rule_engine"
    NODE_JUDGE = "node_judge"
    EDGE_JUDGE = "edge_judge"
    PATH_MATCHER = "path_matcher"
    DOUBLE_CHECK = "double_check"


@dataclass
class Finding:
    """One unit of evidence about anomalous behaviour.

    Attributes
    ----------
    id           : unique within a RiskReport.
    severity     : INFO / WARNING / RISK / CRITICAL.
    category     : NODE / EDGE / ATTACK_PATH.
    code         : short machine-readable identifier
                   (e.g. ``"EDGE_UNAUTHORIZED_TOOL"``).
    message      : human-readable summary.
    subject_id   : node_id / edge_id / attack_path_id this concerns.
    confidence   : detector confidence in [0.0, 1.0].
    source       : which detector produced this.
    remediation  : optional short suggestion (tailored remediations also live
                   on the report-level ``remediations`` list).
    evidence     : detector-specific structured evidence.
    metadata     : free-form.
    """

    id: str
    severity: Severity
    category: FindingCategory
    code: str
    message: str
    subject_id: str
    confidence: float = 1.0
    source: FindingSource = FindingSource.RULE_ENGINE
    remediation: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeVerdict:
    """Output of an LLM-as-a-judge call.

    ``risk_score`` 0.0 = safe ... 1.0 = critical.
    ``flags`` are short tags such as ``"jailbreak"``, ``"hallucination"``,
    ``"prompt_injection"``, ``"contract_violation"``.
    """
    risk_score: float
    confidence: float = 0.5
    flags: list[str] = field(default_factory=list)
    reasoning: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Remediation:
    target_id: str          # node_id, edge_id, or "system"
    action: str             # short tag (e.g. "refine_prompt")
    description: str        # actionable guidance
    priority: Severity = Severity.WARNING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskReport:
    """End-to-end Phase-2 output for one analysed dynamic graph."""

    trace_id: str
    findings: list[Finding] = field(default_factory=list)
    remediations: list[Remediation] = field(default_factory=list)
    origin_nodes: list[str] = field(default_factory=list)
    propagation_paths: list[list[str]] = field(default_factory=list)
    summary: str = ""
    overall_severity: Severity = Severity.INFO
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- queries -----------------------------------------------------------

    def has_critical(self) -> bool:
        return any(f.severity == Severity.CRITICAL for f in self.findings)

    def has_anomaly(self) -> bool:
        return any(
            f.severity in (Severity.RISK, Severity.CRITICAL) for f in self.findings
        )

    def by_category(self, category: FindingCategory) -> list[Finding]:
        return [f for f in self.findings if f.category == category]

    def by_source(self, source: FindingSource) -> list[Finding]:
        return [f for f in self.findings if f.source == source]

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    # -- serialisation -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "overall_severity": self.overall_severity.value,
            "summary": self.summary,
            "findings": [
                {
                    "id": f.id,
                    "severity": f.severity.value,
                    "category": f.category.value,
                    "code": f.code,
                    "message": f.message,
                    "subject_id": f.subject_id,
                    "confidence": f.confidence,
                    "source": f.source.value,
                    "remediation": f.remediation,
                    "evidence": f.evidence,
                }
                for f in self.findings
            ],
            "remediations": [
                {
                    "target_id": r.target_id,
                    "action": r.action,
                    "description": r.description,
                    "priority": r.priority.value,
                }
                for r in self.remediations
            ],
            "origin_nodes": list(self.origin_nodes),
            "propagation_paths": [list(p) for p in self.propagation_paths],
            "generated_at": self.generated_at.isoformat(),
        }
