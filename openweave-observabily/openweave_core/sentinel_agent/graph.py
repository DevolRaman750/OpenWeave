"""
graph.py — InteractionGraph container shared by static + dynamic graphs.

Holds typed nodes, typed edges, adjacency lists, and (for static graphs) the
authorization boundary tables. Both Phase-1 outputs (StaticGraph and
DynamicGraph) are instances of this class — the only difference is what's
populated and what `kind` is set to.

Phase-2 graph comparison will read these structures directly to flag
unauthorised tool calls, illegal agent-to-agent messages, and structurally
unexpected subgraphs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from openweave_core.sentinel_agent.schema import (
    AgentNode,
    Edge,
    EdgeType,
    Node,
    NodeType,
    ToolNode,
)


GraphKind = str  # "static" | "dynamic"


@dataclass
class InteractionGraph:
    """Directed graph of typed Agent/Tool nodes and timestamped Edges.

    Adjacency is materialised as id→[edge_id] lists in both directions, so
    constant-time outgoing/incoming lookups are possible without scanning.

    Attributes
    ----------
    kind                 : ``"static"`` or ``"dynamic"`` — informational only.
    nodes                : node_id → AgentNode | ToolNode.
    edges                : edge_id → Edge.

    Authorisation boundary (typically populated only for static graphs;
    Phase-2 reads these to check dynamic-graph edges against intent):
        allowed_tool_calls : agent_id → set of tool_ids it may invoke
        allowed_messages   : agent_id → set of agent_ids it may message
        authorized_callers : tool_id  → set of agent_ids permitted to call it
    """

    kind: GraphKind = "dynamic"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    # Adjacency lists (private — accessed through helper methods).
    _outgoing: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _incoming: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Static-graph authorization tables.
    allowed_tool_calls: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    allowed_messages: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    authorized_callers: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    # -----------------------------------------------------------------
    # Node operations
    # -----------------------------------------------------------------

    def add_node(self, node: Node) -> Node:
        """Insert *node* (or merge metadata into an existing node of same id).

        Same-id collisions of *different* node types raise ``ValueError`` —
        an agent id reused for a tool is almost certainly a config bug.
        """
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node

        if existing.node_type != node.node_type:
            raise ValueError(
                f"Node id {node.id!r} already exists as {existing.node_type.value}; "
                f"attempted to add as {node.node_type.value}."
            )

        # Same-type re-add: merge metadata (newcomer wins on key collisions).
        existing.metadata.update(node.metadata)
        return existing

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def agents(self) -> Iterator[AgentNode]:
        for n in self.nodes.values():
            if isinstance(n, AgentNode):
                yield n

    def tools(self) -> Iterator[ToolNode]:
        for n in self.nodes.values():
            if isinstance(n, ToolNode):
                yield n

    # -----------------------------------------------------------------
    # Edge operations
    # -----------------------------------------------------------------

    def add_edge(self, edge: Edge) -> Edge:
        """Insert *edge*. Both endpoints must already exist as nodes.

        Re-adding an edge with the same id is treated as an update — the
        adjacency lists are not duplicated.
        """
        if edge.source not in self.nodes:
            raise KeyError(
                f"Edge {edge.id}: source node {edge.source!r} not in graph"
            )
        if edge.target not in self.nodes:
            raise KeyError(
                f"Edge {edge.id}: target node {edge.target!r} not in graph"
            )

        first_insert = edge.id not in self.edges
        self.edges[edge.id] = edge
        if first_insert:
            self._outgoing[edge.source].append(edge.id)
            self._incoming[edge.target].append(edge.id)
        return edge

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [self.edges[eid] for eid in self._outgoing.get(node_id, [])]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [self.edges[eid] for eid in self._incoming.get(node_id, [])]

    def neighbors(self, node_id: str) -> list[str]:
        """Unique downstream node ids reachable in one hop from *node_id*."""
        return list({self.edges[eid].target for eid in self._outgoing.get(node_id, [])})

    def edges_chronological(self) -> list[Edge]:
        """All edges sorted by timestamp ascending — handy for the dynamic graph."""
        return sorted(self.edges.values(), key=lambda e: e.timestamp)

    # -----------------------------------------------------------------
    # Authorization helpers (read against the static-graph boundary)
    # -----------------------------------------------------------------

    def is_tool_call_allowed(self, agent_id: str, tool_id: str) -> bool:
        """True iff the static spec permits *agent_id* to call *tool_id*."""
        return (
            tool_id in self.allowed_tool_calls.get(agent_id, ())
            or agent_id in self.authorized_callers.get(tool_id, ())
        )

    def is_message_allowed(self, sender_id: str, receiver_id: str) -> bool:
        """True iff the static spec permits *sender_id* to message *receiver_id*."""
        return receiver_id in self.allowed_messages.get(sender_id, ())

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        agents = sum(1 for _ in self.agents())
        tools = sum(1 for _ in self.tools())
        edge_counts: dict[str, int] = defaultdict(int)
        for e in self.edges.values():
            edge_counts[e.edge_type.value] += 1
        return {
            "kind": self.kind,
            "agents": agents,
            "tools": tools,
            "edges_total": len(self.edges),
            **{f"edges_{k}": v for k, v in edge_counts.items()},
        }

    def __repr__(self) -> str:  # pragma: no cover
        s = self.summary()
        return (
            f"InteractionGraph(kind={s['kind']!r}, agents={s['agents']}, "
            f"tools={s['tools']}, edges={s['edges_total']})"
        )


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

def merge(*graphs: InteractionGraph, kind: GraphKind = "dynamic") -> InteractionGraph:
    """Union of multiple graphs — useful for merging shards from parallel workers.

    Node-id collisions across graphs must agree on node_type (same rule as
    ``InteractionGraph.add_node``); edge-id collisions are last-write-wins.
    """
    out = InteractionGraph(kind=kind)
    for g in graphs:
        for node in g.nodes.values():
            out.add_node(node)
        for edge in g.edges.values():
            out.add_edge(edge)
        for agent_id, tools in g.allowed_tool_calls.items():
            out.allowed_tool_calls[agent_id].update(tools)
        for agent_id, peers in g.allowed_messages.items():
            out.allowed_messages[agent_id].update(peers)
        for tool_id, callers in g.authorized_callers.items():
            out.authorized_callers[tool_id].update(callers)
    return out
