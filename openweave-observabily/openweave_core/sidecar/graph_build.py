"""
graph_build.py — Build the abstracted graph the UI renders from a TraceBundle.

The Graph View (Phase 4) renders an *abstracted* agent/tool interaction graph
(not the raw span tree): nodes are agents/tools, edges fold every repeated
traversal into one weighted edge (``weight`` = how many times that interaction
fired). A non-converging retrieval loop therefore shows up as a single edge with
``weight = 5`` rather than five parallel edges — exactly the visual signal the
Phase-4 design called for ("fold edge weight into ``graph.edges[].weight``").

We reuse sentinel's ``InteractionGraph`` (cached on
``bundle.extras['dynamic_graph']`` by the sentinel detector) as the source of
truth so the graph's node ids line up with the node/edge ids that sentinel
``NormalizedFlag``s reference. A ``GraphContext`` is returned alongside the
payload so :mod:`openweave_core.sidecar.mapping` can translate every flag's
``subject_id`` onto the matching graph node/edge (see ``graph_subject_id``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from openweave_core.anomaly_pipeline.contracts import TraceBundle
from openweave_core.sentinel_agent.dynamic_graph import (
    SpanResolver,
    build_dynamic_graph,
)
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import NodeType


@dataclass
class GraphContext:
    """Lookups used to align flag/incident subject ids with graph elements."""

    node_ids: set[str] = field(default_factory=set)
    # edge id (== originating span id) -> (source_node_id, target_node_id)
    edge_endpoints: dict[str, tuple[str, str]] = field(default_factory=dict)
    # span id -> the node (agent/tool) under which the span executed
    span_to_node: dict[str, str] = field(default_factory=dict)

    def node_label(self, node_id: str) -> Optional[str]:
        return node_id if node_id in self.node_ids else None


def _node_label(node: Any) -> str:
    # ToolNode has .name; AgentNode falls back to role/id.
    name = getattr(node, "name", None)
    if name:
        return str(name)
    role = getattr(node, "role", None)
    return f"{role}:{node.id}" if role and role != "agent" else node.id


def build_graph(bundle: TraceBundle) -> tuple[dict[str, Any], GraphContext]:
    """Return ``(graph_payload, context)`` for *bundle*.

    ``graph_payload`` is ``{"nodes": [...], "edges": [...]}`` in the exact shape
    ``web/.../utils/graphOverlay.ts::parseGraph`` expects. ``context`` lets the
    mapper resolve detector subject ids onto graph nodes/edges.
    """
    graph: InteractionGraph = bundle.extras.get("dynamic_graph")
    if not isinstance(graph, InteractionGraph):
        graph = build_dynamic_graph(bundle.spans)

    # --- nodes ----------------------------------------------------------
    nodes: list[dict[str, Any]] = []
    for node in graph.nodes.values():
        ntype = (
            "tool"
            if getattr(node, "node_type", None) == NodeType.TOOL
            else "agent"
        )
        nodes.append({"id": node.id, "label": _node_label(node), "type": ntype})

    # --- edges: fold repeated (source,target,type) into one weighted edge
    folded: dict[tuple[str, str, str], dict[str, Any]] = {}
    edge_endpoints: dict[str, tuple[str, str]] = {}
    for edge in graph.edges.values():
        etype = getattr(edge.edge_type, "value", str(edge.edge_type))
        edge_endpoints[edge.id] = (edge.source, edge.target)
        key = (edge.source, edge.target, etype)
        slot = folded.get(key)
        if slot is None:
            folded[key] = {
                "id": f"edge:{edge.source}->{edge.target}",
                "source": edge.source,
                "target": edge.target,
                "type": etype,
                "weight": 1,
            }
        else:
            slot["weight"] += 1
    edges = list(folded.values())

    # --- span -> owning node (target of the span's edge) ----------------
    resolver = SpanResolver()
    span_index = bundle.span_by_id or {s.id: s for s in bundle.spans}
    span_to_node: dict[str, str] = {}
    for span in bundle.spans:
        tool_id = resolver.resolve_tool_id(span)
        node_id = tool_id or resolver.resolve_agent_id(span, span_index)
        span_to_node[span.id] = node_id

    ctx = GraphContext(
        node_ids={n["id"] for n in nodes},
        edge_endpoints=edge_endpoints,
        span_to_node=span_to_node,
    )
    return {"nodes": nodes, "edges": edges}, ctx


def graph_subject_id(raw: str, subject_type: str, ctx: GraphContext) -> str:
    """Translate a detector ``subject_id`` into a graph-aligned subject string.

    Output convention matches ``graphOverlay.ts::parseSubjectId``:
      * ``"node:<id>"``  -> highlights that node
      * ``"edge:<src>-><tgt>"`` -> highlights that edge
    Detectors emit *bare* ids (agent/tool id for nodes, originating span id for
    edges, span id(s) for span/span_pair). We normalise them here so flags and
    incidents address the same nodes/edges the graph renders.
    """
    st = (subject_type or "").lower()

    if st == "node":
        return f"node:{raw}"

    if st == "edge":
        ends = ctx.edge_endpoints.get(raw)
        if ends:
            return f"edge:{ends[0]}->{ends[1]}"
        return f"node:{raw}"  # fall back to a node highlight

    if st in ("span", "span_pair"):
        span_id = raw.split("::", 1)[0] if st == "span_pair" else raw
        node_id = ctx.span_to_node.get(span_id)
        if node_id:
            return f"node:{node_id}"
        return f"span:{raw}"

    # attack_path / trace / unknown — leave addressable but un-prefixed.
    return raw
