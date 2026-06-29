"""
enrichment.py — Resolve every NormalizedFlag's subject into rich context.

Inputs
------
    TraceAnomalyReport.flags  (raw normalized flags)
    TraceBundle               (spans + indexes + envelope)

Outputs
-------
    list[EnrichedFlag]        (one per input flag, in the same order)

Resolution rules
----------------
* ``subject_type == "span"``        → bundle.span_by_id[subject_id]
* ``subject_type == "span_pair"``   → split "a::b"; both spans resolved
* ``subject_type == "edge"``        → sentinel dynamic_graph.edges[id]; if the
                                       edge carries a ``span_id`` we resolve
                                       that span too
* ``subject_type == "node"``        → dynamic_graph.get_node(id)
* ``subject_type == "attack_path"`` → no single subject; we collect span_ids
                                       from the path's evidence dict
* ``subject_type == "trace"``       → no per-span context attached

The sentinel runner caches the dynamic graph on ``bundle.extras['dynamic_graph']``,
so the enricher reads it from there without rebuilding.
"""

from __future__ import annotations

import re
from typing import Optional

from openweave_core.anomaly_pipeline.contracts import (
    NormalizedFlag,
    SUBJECT_ATTACK_PATH,
    SUBJECT_EDGE,
    SUBJECT_NODE,
    SUBJECT_SPAN,
    SUBJECT_SPAN_PAIR,
    SUBJECT_TRACE,
    TraceAnomalyReport,
    TraceBundle,
)
from openweave_core.models.span import ParsedSpan
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    Edge,
    Node,
    ToolNode,
)

from openweave_core.incident_classification.contracts import EnrichedFlag


# Tool-name tokens implying RAG / retrieval semantics. We split the candidate
# name on any non-alphanumeric run (so "tool-vector_search" → ["tool",
# "vector", "search"]) and check token membership — robust to underscores,
# hyphens, colons, slashes alike. Using `\b` boundaries would *miss*
# "vector_search" because `_` is a word character in Python's regex.
_RETRIEVAL_TOKENS: frozenset[str] = frozenset({
    "search", "retrieve", "retriever", "retrieval",
    "lookup", "fetch", "recall", "knn",
    "vector", "embed", "embedding", "embeddings",
    "kb", "knowledge", "context", "document", "doc", "docs",
    "rag", "rerank", "reranker", "index",
    "chunk", "chunks", "grounding",
})
_NON_ALNUM_SPLIT = re.compile(r"[^a-z0-9]+")


def _looks_like_retrieval(name: Optional[str]) -> bool:
    if not name:
        return False
    tokens = _NON_ALNUM_SPLIT.split(name.lower())
    return any(t in _RETRIEVAL_TOKENS for t in tokens)


class FlagEnricher:
    """Resolve subjects and surrounding context for normalized flags."""

    def __init__(self, bundle: TraceBundle) -> None:
        self._bundle = bundle
        self._graph: Optional[InteractionGraph] = bundle.extras.get(
            "dynamic_graph"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, flag: NormalizedFlag) -> EnrichedFlag:
        e = EnrichedFlag(flag=flag)

        st = flag.subject_type
        if st == SUBJECT_SPAN:
            self._attach_span_context(e, flag.subject_id)
        elif st == SUBJECT_SPAN_PAIR:
            self._attach_pair_context(e, flag.subject_id)
        elif st == SUBJECT_EDGE:
            self._attach_edge_context(e, flag.subject_id)
        elif st == SUBJECT_NODE:
            self._attach_node_context(e, flag.subject_id)
        elif st == SUBJECT_ATTACK_PATH:
            self._attach_attack_path_context(e, flag)
        elif st == SUBJECT_TRACE:
            pass  # nothing per-span to attach
        # fall through — unknown subject types behave like SUBJECT_TRACE

        # Derived flags + payload text aggregation
        self._compute_derived(e)
        self._aggregate_payload_text(e)
        # affected_span_ids always includes the primary span / pair / edge span
        self._compute_affected_spans(e)

        return e

    def enrich_all(self, report: TraceAnomalyReport) -> list[EnrichedFlag]:
        return [self.enrich(f) for f in report.flags]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _attach_span_context(self, e: EnrichedFlag, span_id: str) -> None:
        span = self._bundle.span_by_id.get(span_id)
        if span is None:
            return
        e.span = span
        e.parent_span = self._bundle.parent_of(span_id)
        e.children_spans = self._bundle.children_of(span_id)
        e.sibling_spans = self._bundle.siblings_of(span_id)
        e.root_span = self._bundle.root_of(span_id)

    def _attach_pair_context(self, e: EnrichedFlag, pair_id: str) -> None:
        """SPAN_PAIR ids are formatted ``"<id_a>::<id_b>"`` by the cycle adapter."""
        if "::" not in pair_id:
            return
        a_id, b_id = pair_id.split("::", 1)
        e.span = self._bundle.span_by_id.get(a_id)
        e.paired_span = self._bundle.span_by_id.get(b_id)
        if e.span is not None:
            e.parent_span = self._bundle.parent_of(a_id)
            e.sibling_spans = self._bundle.siblings_of(a_id)
            e.root_span = self._bundle.root_of(a_id)

    def _attach_edge_context(self, e: EnrichedFlag, edge_id: str) -> None:
        if self._graph is None:
            return
        edge: Optional[Edge] = self._graph.edges.get(edge_id)
        if edge is None:
            return
        e.edge = edge
        # The dynamic-graph builder maps spans 1:1 to edges, so edge.id is
        # typically the span_id. Try edge_id directly, then edge.span_id.
        span_id = edge.span_id or edge_id
        span = self._bundle.span_by_id.get(span_id)
        if span is not None:
            self._attach_span_context(e, span_id)

    def _attach_node_context(self, e: EnrichedFlag, node_id: str) -> None:
        if self._graph is None:
            return
        node: Optional[Node] = self._graph.get_node(node_id)
        if node is None:
            return
        e.node = node
        # Pull every span the node owns (agent: all of its outgoing edges'
        # span ids; tool: all incoming TOOL_INVOCATION span ids). This lets
        # the correlator group node-flags with span-flags that share scope.
        owned_span_ids: set[str] = set()
        for edge in self._graph.outgoing_edges(node_id):
            if edge.span_id:
                owned_span_ids.add(edge.span_id)
        for edge in self._graph.incoming_edges(node_id):
            if edge.span_id:
                owned_span_ids.add(edge.span_id)
        e.affected_span_ids.update(owned_span_ids)

    def _attach_attack_path_context(
        self, e: EnrichedFlag, flag: NormalizedFlag
    ) -> None:
        """Attack-path subjects don't map to a single span. Pull span ids
        from the flag's structured evidence so correlation still works."""
        evidence = flag.evidence or {}
        for key in ("edge_ids", "origin_nodes", "matched_edge_ids"):
            for sid in evidence.get(key, []) or []:
                if isinstance(sid, str) and sid in self._bundle.span_by_id:
                    e.affected_span_ids.add(sid)
        # First implicated span (if any) becomes the local centre.
        for sid in evidence.get("edge_ids", []) or []:
            if isinstance(sid, str):
                span = self._bundle.span_by_id.get(sid)
                if span:
                    self._attach_span_context(e, sid)
                    break

    # ------------------------------------------------------------------
    # Derived attributes
    # ------------------------------------------------------------------

    def _compute_derived(self, e: EnrichedFlag) -> None:
        tool_name: Optional[str] = None
        is_tool = False
        is_generation = False

        if isinstance(e.node, ToolNode):
            tool_name = e.node.name or e.node.id
            is_tool = True
        elif isinstance(e.node, AgentNode):
            # Agent nodes are never tools, never generations.
            pass

        if e.edge is not None:
            etype = e.edge.edge_type.value if e.edge.edge_type else ""
            if etype == "tool_invocation":
                is_tool = True
                tool_name = tool_name or e.edge.target
            if etype == "reasoning":
                is_generation = True

        if e.span is not None:
            stype = (e.span.span_type or "").upper()
            if stype == "GENERATION":
                is_generation = True
            if e.span.tool_name and not tool_name:
                # The dynamic-graph tool-id normaliser strips "tool-" prefix;
                # do the same here for the derived label.
                lowered = e.span.tool_name.lower()
                tool_name = _strip_tool_prefix(lowered)
                if lowered.startswith(("tool-", "tool:", "tool/")):
                    is_tool = True

        if e.paired_span is not None:
            # Cycle pairs from the cycle detector are by construction
            # tool-related when their parent has a tool-invocation child.
            sp = e.paired_span
            if (sp.tool_name or "").lower().startswith(("tool-", "tool:", "tool/")):
                is_tool = True
                tool_name = tool_name or _strip_tool_prefix(sp.tool_name.lower())

        e.tool_name = tool_name
        e.is_tool = is_tool
        e.is_generation = is_generation
        e.is_retrieval = _looks_like_retrieval(tool_name)
        if not e.is_retrieval and e.span is not None:
            e.is_retrieval = _looks_like_retrieval(e.span.tool_name)
        if not e.is_retrieval and e.paired_span is not None:
            e.is_retrieval = _looks_like_retrieval(e.paired_span.tool_name)

    def _aggregate_payload_text(self, e: EnrichedFlag) -> None:
        texts: list[str] = []

        def _push(s: Optional[str]) -> None:
            if s:
                texts.append(s)

        _push(e.flag.message)
        if e.span is not None:
            _push(e.span.input_text)
            _push(e.span.output_text)
        if e.paired_span is not None:
            _push(e.paired_span.input_text)
            _push(e.paired_span.output_text)
        if e.edge is not None:
            _push(e.edge.message_content)
            _push(e.edge.agent_thoughts)
            if e.edge.tool_args:
                _push(str(e.edge.tool_args))
            if e.edge.response_payload:
                _push(str(e.edge.response_payload))

        e.payload_texts = texts

    def _compute_affected_spans(self, e: EnrichedFlag) -> None:
        if e.span is not None:
            e.affected_span_ids.add(e.span.id)
        if e.paired_span is not None:
            e.affected_span_ids.add(e.paired_span.id)
        if e.edge is not None and e.edge.span_id:
            e.affected_span_ids.add(e.edge.span_id)


def _strip_tool_prefix(name: str) -> str:
    for prefix in ("tool-", "tool:", "tool/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name
