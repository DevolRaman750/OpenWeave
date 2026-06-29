"""
attack_paths.py — Library of known risky topologies + their semantic signatures.

An AttackPath couples:
    * an *exemplar* InteractionGraph (the topology to match structurally), and
    * a *semantic signature* — text patterns / flags / structural invariants
      that the live subgraph must also exhibit for the match to count.

The matcher (``path_matcher.py``) consumes these: first checks structural
similarity via the WL × Edge-Distance kernels, then runs ``semantic_match``
to confirm the live edges actually look like the named exploit.

Three built-in patterns are shipped (call ``builtin_library()``):

    REDUNDANT_TOOL_CYCLE     5×(tool-invocation) burst from one agent
                             — the inefficiency loop the CDCS module catches
                             from the agent side.
    UNAUTHORISED_TOOL_USE    agent → tool edge where tool is outside the
                             agent's static permissions.
    PROMPT_INJECTION_CHAIN   suspicious payload markers in a chained
                             agent→agent→tool sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

from openweave_core.sentinel_agent.findings import Severity
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    Edge,
    EdgeType,
    ToolNode,
)


@dataclass
class SemanticSignature:
    """Constraints checked AFTER a structural match passes the kernel threshold.

    Attributes
    ----------
    min_repeats         : require >= N edges of `repeated_edge_type` from one source.
    repeated_edge_type  : edge type whose repetition implies the pattern.
    text_keywords       : if any payload contains *any* of these (case-insensitive
                          substrings), the semantic match is reinforced.
    requires_unauth     : True ⇒ the matched edge must violate static authorisation.
    requires_chain      : True ⇒ matched subgraph must include an A→A→T chain
                          (delegation followed by tool call).
    """
    min_repeats: int = 0
    repeated_edge_type: Optional[EdgeType] = None
    text_keywords: tuple[str, ...] = ()
    requires_unauth: bool = False
    requires_chain: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackPath:
    """One entry in the library: structural exemplar + semantic signature."""
    id: str
    name: str
    description: str
    pattern_graph: InteractionGraph
    signature: SemanticSignature
    severity: Severity
    remediation: str


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

class AttackPathLibrary:
    """In-memory registry of known AttackPath entries.

    Designed to be hot-reloadable — analysts can add / replace / remove paths
    at runtime to teach the system new exploit topologies. Iteration order
    follows insertion.
    """

    def __init__(self, paths: Optional[list[AttackPath]] = None) -> None:
        self._paths: dict[str, AttackPath] = {}
        for p in paths or ():
            self.register(p)

    def register(self, path: AttackPath) -> None:
        if not path.id:
            raise ValueError("AttackPath.id is required")
        self._paths[path.id] = path

    def unregister(self, path_id: str) -> None:
        self._paths.pop(path_id, None)

    def get(self, path_id: str) -> Optional[AttackPath]:
        return self._paths.get(path_id)

    def all(self) -> list[AttackPath]:
        return list(self._paths.values())

    def __iter__(self) -> Iterator[AttackPath]:
        return iter(self._paths.values())

    def __len__(self) -> int:
        return len(self._paths)


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------

# Pattern timestamps are spaced by 1s starting at this anchor. The matcher
# normalises live edges to this same window before computing the
# edge-distance kernel, so the exact value here is irrelevant — only the
# relative spacing matters.
_PATTERN_ANCHOR = datetime(2000, 1, 1)


def _redundant_tool_cycle() -> AttackPath:
    """Agent that pings the same tool 5 times in quick succession.

    Mirrors the test_agent.py redundant-loop scenario.
    """
    g = InteractionGraph(kind="static")
    agent = AgentNode(id="pat_agent", role="agent")
    tool = ToolNode(id="pat_tool", name="tool")
    g.add_node(agent)
    g.add_node(tool)
    for i in range(5):
        g.add_edge(Edge(
            id=f"pat_redundant_{i}",
            source="pat_agent",
            target="pat_tool",
            edge_type=EdgeType.TOOL_INVOCATION,
            timestamp=_PATTERN_ANCHOR + timedelta(seconds=i),
            metadata={"pattern": True},
        ))

    return AttackPath(
        id="redundant_tool_cycle",
        name="Redundant Tool Cycle",
        description=(
            "An agent calls the same tool repeatedly in quick succession "
            "with no intervening reasoning that would justify the repetition."
        ),
        pattern_graph=g,
        signature=SemanticSignature(
            min_repeats=3,
            repeated_edge_type=EdgeType.TOOL_INVOCATION,
        ),
        severity=Severity.RISK,
        remediation=(
            "Add a guard in the agent's planning loop: break out of the cycle "
            "if the same tool returns substantively identical output twice. "
            "Cache tool results for the duration of the user task."
        ),
    )


def _unauthorised_tool_use() -> AttackPath:
    g = InteractionGraph(kind="static")
    g.add_node(AgentNode(id="pat_agent", role="agent"))
    g.add_node(ToolNode(id="pat_unauth_tool", name="unauthorised_tool"))
    g.add_edge(Edge(
        id="pat_unauth_invocation",
        source="pat_agent",
        target="pat_unauth_tool",
        edge_type=EdgeType.TOOL_INVOCATION,
        timestamp=_PATTERN_ANCHOR,
        metadata={"pattern": True},
    ))

    return AttackPath(
        id="unauthorised_tool_use",
        name="Unauthorised Tool Use",
        description=(
            "Agent invokes a tool that is not present in its static "
            "AgentSpec.allowed_tools. High-confidence single-point exploit."
        ),
        pattern_graph=g,
        signature=SemanticSignature(requires_unauth=True),
        severity=Severity.CRITICAL,
        remediation=(
            "Block the runtime invocation; either explicitly grant the "
            "permission in the static spec (if intended) or tighten the "
            "agent's tool-router whitelist."
        ),
    )


def _prompt_injection_chain() -> AttackPath:
    """Agent A delegates to Agent B which then calls a tool — with injection markers."""
    g = InteractionGraph(kind="static")
    g.add_node(AgentNode(id="pat_orchestrator", role="orchestrator"))
    g.add_node(AgentNode(id="pat_worker", role="worker"))
    g.add_node(ToolNode(id="pat_tool", name="tool"))
    g.add_edge(Edge(
        id="pat_delegation",
        source="pat_orchestrator",
        target="pat_worker",
        edge_type=EdgeType.MESSAGE,
        timestamp=_PATTERN_ANCHOR,
        metadata={"pattern": True},
    ))
    g.add_edge(Edge(
        id="pat_chain_invocation",
        source="pat_worker",
        target="pat_tool",
        edge_type=EdgeType.TOOL_INVOCATION,
        timestamp=_PATTERN_ANCHOR + timedelta(seconds=1),
        metadata={"pattern": True},
    ))

    return AttackPath(
        id="prompt_injection_chain",
        name="Prompt Injection Chain",
        description=(
            "Orchestrator forwards an unsanitised user message to a worker "
            "agent, which then executes a tool call carrying injection "
            "markers ('ignore previous instructions', etc.)."
        ),
        pattern_graph=g,
        signature=SemanticSignature(
            requires_chain=True,
            text_keywords=(
                "ignore previous instructions",
                "ignore all instructions",
                "disregard prior",
                "you are now",
                "system prompt:",
            ),
        ),
        severity=Severity.CRITICAL,
        remediation=(
            "Insert an input-sanitisation filter between the orchestrator and "
            "the worker agent; quarantine messages containing role-override "
            "phrases before they reach any tool boundary."
        ),
    )


def builtin_library() -> AttackPathLibrary:
    """Return a fresh ``AttackPathLibrary`` pre-loaded with shipping patterns."""
    return AttackPathLibrary([
        _redundant_tool_cycle(),
        _unauthorised_tool_use(),
        _prompt_injection_chain(),
    ])
