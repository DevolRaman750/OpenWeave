"""
rule_engine.py — Deterministic rule-based classifiers (Module 1, hard rules).

Hard rules — no ambiguity. They are the lowest-cost, highest-precision layer
of the dual-layer evaluator. Every check here cross-references the **static
graph** built in Phase 1 to determine whether the runtime behaviour is
authorised.

Categories implemented
----------------------
* Edge authorisation : agent→tool calls and agent→agent messages must be
                       declared as permitted in the static graph.
* Node provenance    : runtime nodes must be declared in the static graph
                       (an undeclared agent/tool appearing at runtime is a
                       configuration drift signal at best, a foothold at worst).
* Tool contract      : tool outputs are checked against declared
                       ``ToolNode.output_constraints`` (best-effort shape
                       validation — tighten with a real schema validator
                       upstream when richer constraints are configured).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from openweave_core.sentinel_agent.findings import (
    Finding,
    FindingCategory,
    FindingSource,
    Severity,
)
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    Edge,
    EdgeType,
    Node,
    ToolNode,
)


class RuleBasedEvaluator:
    """Deterministic checks against a frozen Static Graph.

    Construct once per static graph; reuse across many dynamic graphs.
    """

    def __init__(self, static_graph: InteractionGraph) -> None:
        if static_graph.kind != "static":
            raise ValueError(
                f"static_graph.kind must be 'static', got {static_graph.kind!r}"
            )
        self._static = static_graph

    @property
    def static_graph(self) -> InteractionGraph:
        return self._static

    # ------------------------------------------------------------------
    # Edge-level rules
    # ------------------------------------------------------------------

    def check_edge_authorization(self, edge: Edge) -> Optional[Finding]:
        """Flag unauthorised tool invocations / agent-to-agent messages."""
        if edge.edge_type is EdgeType.TOOL_INVOCATION:
            if not self._static.is_tool_call_allowed(edge.source, edge.target):
                return Finding(
                    id=f"rule::{edge.id}::unauthorized_tool",
                    severity=Severity.CRITICAL,
                    category=FindingCategory.EDGE,
                    code="EDGE_UNAUTHORIZED_TOOL",
                    message=(
                        f"Agent {edge.source!r} invoked tool {edge.target!r} "
                        "without static-graph permission."
                    ),
                    subject_id=edge.id,
                    confidence=1.0,
                    source=FindingSource.RULE_ENGINE,
                    remediation=(
                        f"Grant {edge.source!r} access to {edge.target!r} in "
                        "AgentSpec.allowed_tools / ToolSpec.authorized_callers, "
                        "OR block this invocation at runtime."
                    ),
                    evidence={
                        "source": edge.source,
                        "target": edge.target,
                        "timestamp": edge.timestamp.isoformat(),
                        "span_id": edge.span_id,
                    },
                )

        elif edge.edge_type is EdgeType.MESSAGE:
            if not self._static.is_message_allowed(edge.source, edge.target):
                return Finding(
                    id=f"rule::{edge.id}::unauthorized_message",
                    severity=Severity.RISK,
                    category=FindingCategory.EDGE,
                    code="EDGE_UNAUTHORIZED_MESSAGE",
                    message=(
                        f"Agent {edge.source!r} sent a message to "
                        f"{edge.target!r} outside the declared peer list."
                    ),
                    subject_id=edge.id,
                    confidence=1.0,
                    source=FindingSource.RULE_ENGINE,
                    remediation=(
                        f"Either declare {edge.target!r} in {edge.source!r}'s "
                        "allowed_peers, or block this message channel."
                    ),
                    evidence={
                        "source": edge.source,
                        "target": edge.target,
                        "timestamp": edge.timestamp.isoformat(),
                    },
                )

        return None

    # ------------------------------------------------------------------
    # Node-level rules
    # ------------------------------------------------------------------

    def check_node_known(self, node: Node) -> Optional[Finding]:
        """Flag runtime nodes absent from the static graph (config drift)."""
        if node.id in self._static.nodes:
            return None
        kind = node.node_type.value
        return Finding(
            id=f"rule::{node.id}::unknown_{kind}",
            severity=Severity.WARNING,
            category=FindingCategory.NODE,
            code=f"NODE_UNKNOWN_{kind.upper()}",
            message=(
                f"Runtime {kind} {node.id!r} is not declared in the static graph."
            ),
            subject_id=node.id,
            confidence=1.0,
            source=FindingSource.RULE_ENGINE,
            remediation=(
                f"Declare {node.id!r} in SystemSpec, or investigate why an "
                "undeclared entity is operating at runtime."
            ),
        )

    def check_agent_uses_unauthorised_tools(
        self, agent: AgentNode, dynamic_graph: InteractionGraph
    ) -> list[Finding]:
        """Aggregate per-agent unauthorised-tool report (across all its edges)."""
        offences: list[str] = []
        for e in dynamic_graph.outgoing_edges(agent.id):
            if e.edge_type is EdgeType.TOOL_INVOCATION and not (
                self._static.is_tool_call_allowed(agent.id, e.target)
            ):
                offences.append(e.target)

        if not offences:
            return []

        unique = sorted(set(offences))
        return [
            Finding(
                id=f"rule::{agent.id}::accessed_unauthorised_tools",
                severity=Severity.CRITICAL,
                category=FindingCategory.NODE,
                code="NODE_AGENT_UNAUTHORIZED_TOOL_USE",
                message=(
                    f"Agent {agent.id!r} invoked tools outside its permission "
                    f"set: {unique}."
                ),
                subject_id=agent.id,
                confidence=1.0,
                source=FindingSource.RULE_ENGINE,
                remediation=(
                    f"Tighten {agent.id!r}'s system prompt and/or revoke runtime "
                    f"access to {unique}."
                ),
                evidence={"unauthorised_tools": unique, "n_attempts": len(offences)},
            )
        ]

    # ------------------------------------------------------------------
    # Tool-output contract validation
    # ------------------------------------------------------------------

    def check_tool_output_contract(
        self, tool_id: str, output: Any, *, edge_id: str = ""
    ) -> Optional[Finding]:
        """Best-effort schema check against ToolNode.output_constraints.

        If ``output`` is a dict and the static tool declares required keys,
        we flag missing keys. Empty / non-dict output with a non-empty
        contract is *not* an automatic violation — many tools legitimately
        return strings. Callers wanting stricter checks should plug in a
        real validator (e.g. jsonschema).
        """
        static_tool = self._static.get_node(tool_id)
        if not isinstance(static_tool, ToolNode):
            return None
        constraints = static_tool.output_constraints or {}
        if not constraints or not isinstance(output, dict):
            return None

        required_keys = list(constraints.keys())
        missing = [k for k in required_keys if k not in output]
        if not missing:
            return None

        return Finding(
            id=f"rule::{edge_id or tool_id}::contract_violation",
            severity=Severity.RISK,
            category=FindingCategory.NODE,
            code="TOOL_OUTPUT_CONTRACT_VIOLATION",
            message=(
                f"Tool {tool_id!r} output missing required keys: {missing}."
            ),
            subject_id=tool_id,
            confidence=0.9,
            source=FindingSource.RULE_ENGINE,
            remediation=(
                f"Fix the {tool_id!r} implementation to return the declared "
                f"output schema, or update the tool's output_constraints."
            ),
            evidence={"missing_keys": missing, "received_keys": list(output.keys())},
        )

    # ------------------------------------------------------------------
    # Convenience: run every rule over a whole dynamic graph
    # ------------------------------------------------------------------

    def evaluate_all(self, dynamic_graph: InteractionGraph) -> list[Finding]:
        out: list[Finding] = []
        for node in dynamic_graph.nodes.values():
            if (f := self.check_node_known(node)):
                out.append(f)
            if isinstance(node, AgentNode):
                out.extend(self.check_agent_uses_unauthorised_tools(node, dynamic_graph))
        for edge in dynamic_graph.edges.values():
            if (f := self.check_edge_authorization(edge)):
                out.append(f)
        return out
