"""
schema.py — Typed nodes / edges / specs for the SentinelAgent interaction graph.

Phase 1 of the SentinelAgent framework models a Multi-Agent System as a
directed graph G = (V, E) where:

    Nodes V : AgentNode  | ToolNode      (typed, with rich metadata)
    Edges E : Edge       (typed interactions, strictly timestamped)

Both the *Static Graph* (built from system configuration) and the *Dynamic
Execution Graph* (built from live traces) share these schemas. That symmetry
is what enables side-by-side comparison in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    AGENT = "agent"
    TOOL = "tool"


class EdgeType(str, Enum):
    """Type of directed interaction recorded as an edge."""

    MESSAGE = "message"                  # agent → agent (data passing)
    TOOL_INVOCATION = "tool_invocation"  # agent → tool  (call)
    TOOL_RESPONSE = "tool_response"      # tool  → agent (result)
    REASONING = "reasoning"              # agent → self  (LLM generation / thought)
    DELEGATION = "delegation"            # agent → agent (hand off control)
    UNKNOWN = "unknown"                  # default when type cannot be inferred


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

@dataclass
class AgentNode:
    """An LLM agent in the system.

    Attributes
    ----------
    id                     : globally unique identifier (within one graph).
    role                   : e.g. ``"orchestrator"``, ``"code_executor"``,
                             ``"summarizer"``.
    system_prompt          : the agent's static instructions / persona.
    declared_capabilities  : list of self-declared capabilities; informational.
    metadata               : free-form properties (model name, version, owner …).
    """

    id: str
    role: str = ""
    system_prompt: str = ""
    declared_capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    node_type: NodeType = field(default=NodeType.AGENT, init=False)


@dataclass
class ToolNode:
    """An external function / API / environment endpoint callable by agents.

    Attributes
    ----------
    id                  : globally unique identifier.
    name                : human-readable name (often same as id).
    description         : what the tool does.
    input_constraints   : declared schema / validation rules for the input
                          (e.g. JSON schema, regex, type spec).
    output_constraints  : declared schema / contract for the output.
    behavioral_contract : free-text or structured spec of expected behaviour
                          (side-effects, idempotency, latency budget, etc.).
    metadata            : free-form properties.
    """

    id: str
    name: str = ""
    description: str = ""
    input_constraints: dict[str, Any] = field(default_factory=dict)
    output_constraints: dict[str, Any] = field(default_factory=dict)
    behavioral_contract: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    node_type: NodeType = field(default=NodeType.TOOL, init=False)


# Union covering both node variants — most graph APIs accept either.
Node = Union[AgentNode, ToolNode]


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """A directed, strictly-timestamped interaction between two nodes.

    A Phase-1 Edge maps directly onto a single ParsedSpan in the dynamic graph;
    in the static graph it represents one allowed interaction path.

    Payload fields are all optional because different edge types carry
    different data — fill what's relevant for the edge_type.

    Attributes
    ----------
    id                : edge identifier; for dynamic edges we typically reuse
                        the originating span_id.
    source            : node_id of the actor (caller).
    target            : node_id of the recipient (callee).
    edge_type         : kind of interaction (see EdgeType).
    timestamp         : when the interaction occurred. Static edges should
                        use a stable sentinel (e.g. ``datetime.min``).

    Payload
    -------
    message_content   : raw message text passed agent → agent.
    tool_args         : structured arguments passed to a tool.
    response_payload  : tool's response data.
    agent_thoughts    : the agent's internal reasoning trace ("thought" text).

    Provenance
    ----------
    span_id, trace_id, parent_edge_id : backward links to the originating span
                        (None for static-graph edges).
    metadata          : free-form attributes (latency, tokens, cost, …).
    """

    id: str
    source: str
    target: str
    edge_type: EdgeType
    timestamp: datetime

    # Payload (any subset may be populated for a given edge_type)
    message_content: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    response_payload: Any = None
    agent_thoughts: Optional[str] = None

    # Provenance
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    parent_edge_id: Optional[str] = None

    # Free-form
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Static-graph specifications (config-level inputs)
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """Configuration entry for one agent in the system specification.

    ``allowed_tools`` and ``allowed_peers`` define the *authorization
    boundary* used to build the static graph's permitted interaction paths.
    """

    id: str
    role: str = ""
    system_prompt: str = ""
    capabilities: list[str] = field(default_factory=list)

    allowed_tools: list[str] = field(default_factory=list)
    allowed_peers: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    """Configuration entry for one tool in the system specification.

    ``authorized_callers`` is the inverse view of agent.allowed_tools — both
    must agree for the static graph to consider the call authorised. The
    static builder will validate / reconcile the two.
    """

    id: str
    name: str = ""
    description: str = ""
    input_constraints: dict[str, Any] = field(default_factory=dict)
    output_constraints: dict[str, Any] = field(default_factory=dict)
    behavioral_contract: str = ""

    authorized_callers: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSpec:
    """Top-level configuration: all agents + all tools for one MAS."""

    agents: list[AgentSpec] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
