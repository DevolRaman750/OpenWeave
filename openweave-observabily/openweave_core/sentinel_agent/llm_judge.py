"""
llm_judge.py — LLM-as-a-judge interfaces and default heuristic implementations.

The Phase-2 dual-layer evaluator pairs deterministic rules (``rule_engine``)
with semantic judges that operate on free-text payloads. Real deployments
plug in:

    * For Node Judging  : IBM Granite Guardian 3.2 (jailbreaks / function-call
                          hallucinations).
    * For Edge Judging  : Llamafirewall (direct / indirect prompt injections in
                          inter-node messages).

This module provides:
    * Two ``Protocol`` types that any external model wrapper must satisfy.
    * ``HeuristicNodeJudge`` / ``HeuristicEdgeJudge`` — lightweight, no-API,
      regex/keyword-based fallbacks that ship enabled by default so the
      pipeline produces real (if conservative) signals out of the box.
    * ``NullNodeJudge`` / ``NullEdgeJudge`` — pass-through judges used when
      the LLM layer is intentionally disabled.

All judges return a ``JudgeVerdict``. Verdict semantics:
    risk_score = 0.0 ......... unequivocally safe / not applicable
    risk_score > 0.7 ......... high-confidence anomaly
    confidence = uncertainty in the risk_score itself
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from openweave_core.sentinel_agent.findings import JudgeVerdict
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    Edge,
    EdgeType,
    Node,
    ToolNode,
)


# ---------------------------------------------------------------------------
# Context objects passed to judges
# ---------------------------------------------------------------------------

@dataclass
class NodeContext:
    """Everything a node judge can look at besides the node itself."""
    static_node: Optional[Node]
    incoming_edges: list[Edge]
    outgoing_edges: list[Edge]
    dynamic_graph: InteractionGraph


@dataclass
class EdgeContext:
    """Everything an edge judge can look at besides the edge itself."""
    source_node: Optional[Node]
    target_node: Optional[Node]
    static_authorized: bool
    dynamic_graph: InteractionGraph


# ---------------------------------------------------------------------------
# Protocols — what an externally-wired LLM judge must implement
# ---------------------------------------------------------------------------

@runtime_checkable
class NodeJudge(Protocol):
    """Internal-integrity judge for a single node.

    For agents: detect jailbreaks, role-violating outputs, hallucinated plans,
    inconsistent reasoning, function-calling hallucinations.
    For tools:  detect output-contract violations, schema drift.
    """

    def evaluate_node(self, node: Node, context: NodeContext) -> JudgeVerdict:
        ...


@runtime_checkable
class EdgeJudge(Protocol):
    """Relational judge for one interaction edge.

    Detect indirect / direct prompt injection in payloads, collusive
    delegation, malicious tool arguments, conversation-misalignment risks.
    """

    def evaluate_edge(self, edge: Edge, context: EdgeContext) -> JudgeVerdict:
        ...


# ---------------------------------------------------------------------------
# Null judges — explicit "I have nothing to say"
# ---------------------------------------------------------------------------

class NullNodeJudge:
    """Returns risk_score=0 with confidence=0 — judge intentionally disabled."""

    def evaluate_node(self, node: Node, context: NodeContext) -> JudgeVerdict:  # noqa: ARG002
        return JudgeVerdict(risk_score=0.0, confidence=0.0, reasoning="null judge")


class NullEdgeJudge:
    def evaluate_edge(self, edge: Edge, context: EdgeContext) -> JudgeVerdict:  # noqa: ARG002
        return JudgeVerdict(risk_score=0.0, confidence=0.0, reasoning="null judge")


# ---------------------------------------------------------------------------
# Heuristic judges — regex/keyword baselines (no LLM API required)
# ---------------------------------------------------------------------------

# Patterns that strongly suggest prompt-injection / jailbreak attempts.
_JAILBREAK_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"ignore (?:all|any|previous|prior|the above) instructions",
        r"disregard (?:all|any|previous|prior) (?:instructions|prompts)",
        r"forget (?:your|all|previous) (?:instructions|guidelines|rules)",
        r"you are now (?:dan|do anything now|jailbroken|unrestricted)",
        r"act as (?:if you (?:are|were) )?(?:not )?(?:an? )?(?:ai|assistant|model)",
        r"system\s*(?:prompt|message)\s*[:=]",
        r"<\s*(?:system|admin)\s*>",
    )
)

# Patterns suggesting hallucinated tool arguments or unsafe payloads.
_TOOL_HALLUCINATION_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"(?:rm|del|format|truncate)\s+(?:-rf?|/)\s*/",   # destructive shell
        r"DROP\s+TABLE\b",
        r"\.\./",                                          # path traversal
        r"file://(?:/etc|/proc|/root|/home)",
        r"(?:powershell|cmd|bash)\s+-e",
    )
)

# Sensitive-content / data-leakage signals.
_LEAKAGE_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p) for p in (
        r"(?i)\b(?:api[_\- ]?key|secret|token|password|bearer)\s*[:=]\s*\S+",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"AKIA[0-9A-Z]{16}",         # AWS access key
    )
)


def _scan(text: str, patterns: tuple[re.Pattern, ...]) -> list[str]:
    if not text:
        return []
    hits: list[str] = []
    for pat in patterns:
        m = pat.search(text)
        if m:
            hits.append(pat.pattern)
    return hits


class HeuristicNodeJudge:
    """Cheap, dependency-free node integrity checks.

    Designed as a sensible default so the pipeline produces real signal even
    when no real LLM judge has been wired in. Swap for IBM Granite Guardian
    (or any other ``NodeJudge``) in production.

    Signals
    -------
    Agent nodes
        * jailbreak / injection markers in the agent's *outgoing* edges
          (the agent itself emitted suspicious text).
        * role/system_prompt absent (configuration drift).
        * unauthorised tool calls already covered by rule engine — we just
          surface a corroborating signal here.
    Tool nodes
        * output contains payloads suggesting hallucinated / unsafe args.
        * declared output_constraints exist but never appear satisfied.
    """

    def evaluate_node(self, node: Node, context: NodeContext) -> JudgeVerdict:
        if isinstance(node, AgentNode):
            return self._evaluate_agent(node, context)
        if isinstance(node, ToolNode):
            return self._evaluate_tool(node, context)
        return JudgeVerdict(risk_score=0.0, confidence=0.1)

    # -- agent -----------------------------------------------------------

    def _evaluate_agent(self, node: AgentNode, ctx: NodeContext) -> JudgeVerdict:
        flags: list[str] = []
        evidence: dict[str, Any] = {}

        all_text: list[str] = []
        for e in ctx.outgoing_edges:
            for piece in (e.message_content, e.agent_thoughts):
                if piece:
                    all_text.append(piece)
        merged = "\n".join(all_text)

        jail = _scan(merged, _JAILBREAK_PATTERNS)
        if jail:
            flags.append("jailbreak_in_output")
            evidence["jailbreak_patterns"] = jail

        if not node.role:
            flags.append("undeclared_role")

        leak = _scan(merged, _LEAKAGE_PATTERNS)
        if leak:
            flags.append("potential_secret_leak")
            evidence["leakage_patterns"] = leak

        # Role/permission inconsistency: declared capabilities vs static node.
        if ctx.static_node is None:
            flags.append("not_in_static_graph")

        risk = _aggregate_risk(flags, severities={
            "jailbreak_in_output": 0.85,
            "potential_secret_leak": 0.95,
            "undeclared_role": 0.25,
            "not_in_static_graph": 0.40,
        })

        return JudgeVerdict(
            risk_score=risk,
            confidence=0.55 if flags else 0.7,
            flags=flags,
            reasoning="heuristic agent scan",
            evidence=evidence,
        )

    # -- tool ------------------------------------------------------------

    def _evaluate_tool(self, node: ToolNode, ctx: NodeContext) -> JudgeVerdict:
        flags: list[str] = []
        evidence: dict[str, Any] = {}

        # Collect every output observed *from* this tool (TOOL_RESPONSE outgoing,
        # or response_payload on incoming TOOL_INVOCATION edges).
        observations: list[str] = []
        for e in ctx.incoming_edges:
            if e.edge_type is EdgeType.TOOL_INVOCATION and e.response_payload:
                observations.append(str(e.response_payload))
        for e in ctx.outgoing_edges:
            if e.edge_type is EdgeType.TOOL_RESPONSE and e.message_content:
                observations.append(e.message_content)
        merged = "\n".join(observations)

        hallu = _scan(merged, _TOOL_HALLUCINATION_PATTERNS)
        if hallu:
            flags.append("unsafe_tool_payload")
            evidence["unsafe_patterns"] = hallu

        leak = _scan(merged, _LEAKAGE_PATTERNS)
        if leak:
            flags.append("tool_output_leak")
            evidence["leakage_patterns"] = leak

        if ctx.static_node is None:
            flags.append("not_in_static_graph")

        risk = _aggregate_risk(flags, severities={
            "unsafe_tool_payload": 0.9,
            "tool_output_leak": 0.95,
            "not_in_static_graph": 0.40,
        })

        return JudgeVerdict(
            risk_score=risk,
            confidence=0.55 if flags else 0.7,
            flags=flags,
            reasoning="heuristic tool scan",
            evidence=evidence,
        )


class HeuristicEdgeJudge:
    """Cheap, dependency-free relational checks.

    Inspects the message_content / tool_args / agent_thoughts payloads for
    indirect-prompt-injection markers, suspicious arguments, and leakage.
    """

    def evaluate_edge(self, edge: Edge, context: EdgeContext) -> JudgeVerdict:
        flags: list[str] = []
        evidence: dict[str, Any] = {}

        # Aggregate every text-bearing slot on the edge.
        text_blob = "\n".join(
            piece for piece in (
                edge.message_content,
                edge.agent_thoughts,
                str(edge.tool_args) if edge.tool_args else "",
                str(edge.response_payload) if edge.response_payload else "",
            ) if piece
        )

        if jail := _scan(text_blob, _JAILBREAK_PATTERNS):
            flags.append("prompt_injection")
            evidence["injection_patterns"] = jail

        if hallu := _scan(text_blob, _TOOL_HALLUCINATION_PATTERNS):
            flags.append("hallucinated_tool_args")
            evidence["unsafe_patterns"] = hallu

        if leak := _scan(text_blob, _LEAKAGE_PATTERNS):
            flags.append("payload_leak")
            evidence["leakage_patterns"] = leak

        # Authorisation already covered by rule engine; surface a
        # corroborating signal here for the judge layer.
        if not context.static_authorized:
            if edge.edge_type in (EdgeType.TOOL_INVOCATION, EdgeType.MESSAGE):
                flags.append("unauthorised_relation")

        risk = _aggregate_risk(flags, severities={
            "prompt_injection": 0.9,
            "hallucinated_tool_args": 0.85,
            "payload_leak": 0.95,
            "unauthorised_relation": 0.7,
        })

        return JudgeVerdict(
            risk_score=risk,
            confidence=0.55 if flags else 0.7,
            flags=flags,
            reasoning="heuristic edge scan",
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_risk(flags: list[str], *, severities: dict[str, float]) -> float:
    """Combine per-flag severities into a single risk_score in [0, 1].

    Uses 1 - prod(1 - s_i) (noisy-OR) so multiple weak signals can still
    aggregate to a high score, but a single strong signal alone is enough.
    """
    if not flags:
        return 0.0
    product = 1.0
    for f in flags:
        s = float(severities.get(f, 0.3))
        product *= (1.0 - s)
    return float(max(0.0, min(1.0, 1.0 - product)))
