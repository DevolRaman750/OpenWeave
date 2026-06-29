"""
status_evaluator.py — Module 1 orchestrators: Node & Edge status evaluation.

The two evaluators here are the *dual-layer* engine described in the
SentinelAgent paper:

    rule layer  : RuleBasedEvaluator (hard, deterministic)
    judge layer : NodeJudge / EdgeJudge (semantic, LLM-driven)

Each evaluator runs both layers and merges their outputs into a flat
``list[Finding]`` for the caller (typically the BehaviorAnalyzer).

The judges are pluggable; the default (HeuristicNodeJudge / HeuristicEdgeJudge)
is regex-based and dependency-free, so the evaluator emits real signal even
without an external LLM wired in.
"""

from __future__ import annotations

from typing import Optional

from openweave_core.sentinel_agent.findings import (
    Finding,
    FindingCategory,
    FindingSource,
    JudgeVerdict,
    Severity,
)
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.llm_judge import (
    EdgeContext,
    EdgeJudge,
    HeuristicEdgeJudge,
    HeuristicNodeJudge,
    NodeContext,
    NodeJudge,
)
from openweave_core.sentinel_agent.rule_engine import RuleBasedEvaluator
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    Edge,
    Node,
    ToolNode,
)

# Default risk thresholds — verdicts ≥ this become a Finding.
DEFAULT_RISK_THRESHOLD: float = 0.5
# In the Module-2 "double check" path the threshold is lowered to this:
STRICT_RISK_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Node evaluator
# ---------------------------------------------------------------------------

class NodeEvaluator:
    """Run rules + LLM-as-a-judge on every node of a dynamic graph."""

    def __init__(
        self,
        *,
        rule_engine: RuleBasedEvaluator,
        node_judge: Optional[NodeJudge] = None,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    ) -> None:
        self._rules = rule_engine
        self._judge: NodeJudge = node_judge or HeuristicNodeJudge()
        self._threshold = risk_threshold

    @property
    def risk_threshold(self) -> float:
        return self._threshold

    def evaluate(
        self,
        node: Node,
        dynamic_graph: InteractionGraph,
        *,
        threshold_override: Optional[float] = None,
    ) -> list[Finding]:
        out: list[Finding] = []

        # Rule layer
        if (f := self._rules.check_node_known(node)):
            out.append(f)
        if isinstance(node, AgentNode):
            out.extend(
                self._rules.check_agent_uses_unauthorised_tools(node, dynamic_graph)
            )

        # Judge layer
        ctx = NodeContext(
            static_node=self._rules.static_graph.get_node(node.id),
            incoming_edges=dynamic_graph.incoming_edges(node.id),
            outgoing_edges=dynamic_graph.outgoing_edges(node.id),
            dynamic_graph=dynamic_graph,
        )
        verdict = self._judge.evaluate_node(node, ctx)
        threshold = threshold_override if threshold_override is not None else self._threshold
        if verdict.risk_score >= threshold:
            out.append(self._verdict_to_finding(node, verdict, threshold_override is not None))

        return out

    def evaluate_all(
        self,
        dynamic_graph: InteractionGraph,
        *,
        threshold_override: Optional[float] = None,
    ) -> list[Finding]:
        return [
            f
            for node in dynamic_graph.nodes.values()
            for f in self.evaluate(node, dynamic_graph, threshold_override=threshold_override)
        ]

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _verdict_to_finding(
        self, node: Node, verdict: JudgeVerdict, double_check: bool
    ) -> Finding:
        sev = _severity_from_risk(verdict.risk_score)
        kind = node.node_type.value
        source = (
            FindingSource.DOUBLE_CHECK if double_check else FindingSource.NODE_JUDGE
        )
        rem = _node_remediation(node, verdict)
        return Finding(
            id=f"judge::{node.id}::{kind}_integrity",
            severity=sev,
            category=FindingCategory.NODE,
            code=f"NODE_JUDGE_{kind.upper()}",
            message=(
                f"Judge flagged {kind} {node.id!r} (risk={verdict.risk_score:.2f}, "
                f"flags={verdict.flags or 'n/a'})."
            ),
            subject_id=node.id,
            confidence=verdict.confidence,
            source=source,
            remediation=rem,
            evidence={
                "risk_score": verdict.risk_score,
                "flags": verdict.flags,
                "reasoning": verdict.reasoning,
                "judge_evidence": verdict.evidence,
                "double_check": double_check,
            },
        )


# ---------------------------------------------------------------------------
# Edge evaluator
# ---------------------------------------------------------------------------

class EdgeEvaluator:
    """Run rules + LLM-as-a-judge on every edge of a dynamic graph."""

    def __init__(
        self,
        *,
        rule_engine: RuleBasedEvaluator,
        edge_judge: Optional[EdgeJudge] = None,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    ) -> None:
        self._rules = rule_engine
        self._judge: EdgeJudge = edge_judge or HeuristicEdgeJudge()
        self._threshold = risk_threshold

    @property
    def risk_threshold(self) -> float:
        return self._threshold

    def evaluate(
        self,
        edge: Edge,
        dynamic_graph: InteractionGraph,
        *,
        threshold_override: Optional[float] = None,
    ) -> list[Finding]:
        out: list[Finding] = []

        # Rule layer
        if (f := self._rules.check_edge_authorization(edge)):
            out.append(f)

        # Judge layer
        src = dynamic_graph.get_node(edge.source)
        tgt = dynamic_graph.get_node(edge.target)
        static_g = self._rules.static_graph
        is_auth = (
            static_g.is_tool_call_allowed(edge.source, edge.target)
            or static_g.is_message_allowed(edge.source, edge.target)
        )
        ctx = EdgeContext(
            source_node=src,
            target_node=tgt,
            static_authorized=is_auth,
            dynamic_graph=dynamic_graph,
        )
        verdict = self._judge.evaluate_edge(edge, ctx)
        threshold = threshold_override if threshold_override is not None else self._threshold
        if verdict.risk_score >= threshold:
            out.append(self._verdict_to_finding(edge, verdict, threshold_override is not None))

        return out

    def evaluate_all(
        self,
        dynamic_graph: InteractionGraph,
        *,
        threshold_override: Optional[float] = None,
        only_edge_ids: Optional[set[str]] = None,
    ) -> list[Finding]:
        edges = (
            (dynamic_graph.edges[eid] for eid in only_edge_ids if eid in dynamic_graph.edges)
            if only_edge_ids is not None
            else dynamic_graph.edges.values()
        )
        return [
            f
            for edge in edges
            for f in self.evaluate(edge, dynamic_graph, threshold_override=threshold_override)
        ]

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _verdict_to_finding(
        self, edge: Edge, verdict: JudgeVerdict, double_check: bool
    ) -> Finding:
        sev = _severity_from_risk(verdict.risk_score)
        source = (
            FindingSource.DOUBLE_CHECK if double_check else FindingSource.EDGE_JUDGE
        )
        rem = _edge_remediation(edge, verdict)
        return Finding(
            id=f"judge::{edge.id}::edge_integrity",
            severity=sev,
            category=FindingCategory.EDGE,
            code="EDGE_JUDGE",
            message=(
                f"Judge flagged edge {edge.source!r} -> {edge.target!r} "
                f"(risk={verdict.risk_score:.2f}, flags={verdict.flags or 'n/a'})."
            ),
            subject_id=edge.id,
            confidence=verdict.confidence,
            source=source,
            remediation=rem,
            evidence={
                "risk_score": verdict.risk_score,
                "flags": verdict.flags,
                "reasoning": verdict.reasoning,
                "judge_evidence": verdict.evidence,
                "edge_type": edge.edge_type.value,
                "double_check": double_check,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity_from_risk(risk: float) -> Severity:
    if risk >= 0.85:
        return Severity.CRITICAL
    if risk >= 0.5:
        return Severity.RISK
    if risk >= 0.3:
        return Severity.WARNING
    return Severity.INFO


def _node_remediation(node: Node, verdict: JudgeVerdict) -> Optional[str]:
    if not verdict.flags:
        return None
    if "jailbreak_in_output" in verdict.flags or "prompt_injection" in verdict.flags:
        return (
            f"Refine the system prompt for {node.id!r}; add a defensive "
            "preamble that rejects 'ignore previous instructions' attempts."
        )
    if "unsafe_tool_payload" in verdict.flags:
        return (
            f"Constrain {node.id!r}'s output schema; reject shell-like / "
            "filesystem-traversal payloads at the response boundary."
        )
    if "potential_secret_leak" in verdict.flags or "tool_output_leak" in verdict.flags:
        return (
            f"Add a secret-scrubbing post-processor in front of {node.id!r}'s "
            "outputs and rotate any leaked credentials immediately."
        )
    if "not_in_static_graph" in verdict.flags:
        return (
            f"Declare {node.id!r} in SystemSpec or investigate why an "
            "undeclared entity is operating at runtime."
        )
    return f"Investigate flags: {verdict.flags}."


def _edge_remediation(edge: Edge, verdict: JudgeVerdict) -> Optional[str]:
    if not verdict.flags:
        return None
    if "prompt_injection" in verdict.flags:
        return (
            f"Add an input-sanitisation filter between {edge.source!r} and "
            f"{edge.target!r}; quarantine messages containing override phrases."
        )
    if "hallucinated_tool_args" in verdict.flags:
        return (
            f"Validate tool arguments against the declared input schema "
            f"before {edge.target!r} executes; reject path-traversal / "
            "shell-injection patterns."
        )
    if "payload_leak" in verdict.flags:
        return (
            "Strip secrets from payloads at the interaction boundary; rotate "
            "any leaked credentials immediately."
        )
    if "unauthorised_relation" in verdict.flags:
        return (
            f"Either grant {edge.source!r} explicit permission for this "
            "interaction, or block it at the message bus / tool router."
        )
    return f"Investigate flags: {verdict.flags}."
