"""
dynamic_graph.py — Event Monitor: build the Dynamic Execution Graph from live
                   ParsedSpan traces.

The EventMonitor restructures flat span logs into typed AgentNode / ToolNode
nodes and timestamped Edges. Mapping rules are encapsulated in a pluggable
``SpanResolver`` so callers can override the defaults to suit their tracing
conventions.

Default resolution heuristics (Langfuse-flavoured)
--------------------------------------------------
* Agent identity for a span = name of its *nearest ancestor whose own
  parent_id is None* (i.e. the root span of the trace). Falls back to a
  ``trace_id``-derived id if no root is found.
* A span is a tool invocation when **any** of:
    - its ``span_type`` is "TOOL"           (Langfuse "TOOL" observation), or
    - its ``tool_name`` starts with "tool-" or "tool:", or
    - its ``metadata`` carries an explicit ``tool_id`` / ``tool`` key.
* Otherwise:
    - ``span_type == "GENERATION"`` ⇒ REASONING edge (agent → self, an LLM call).
    - Anything else ⇒ UNKNOWN edge (recorded but not type-classified).

These rules cover the common case and degrade gracefully on unfamiliar
traces. Override ``SpanResolver`` for custom MAS conventions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

from openweave_core.models.span import ParsedSpan
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    Edge,
    EdgeType,
    ToolNode,
)

log = logging.getLogger(__name__)

# Callable that returns a snapshot of current system state to attach to edges.
SystemStateProvider = Callable[[ParsedSpan], dict[str, Any]]


# ---------------------------------------------------------------------------
# Pluggable span resolver
# ---------------------------------------------------------------------------

class SpanResolver:
    """Maps ParsedSpan → (agent_id, tool_id?, edge_type).

    Subclass and override individual methods to adapt to a different tracing
    convention. The defaults target Langfuse-style traces produced by our
    pipeline (see ``openweave_core.parser.trace_fetcher``).
    """

    # -- agent resolution --------------------------------------------------

    def resolve_agent_id(
        self,
        span: ParsedSpan,
        span_index: dict[str, ParsedSpan],
    ) -> str:
        """Return the id of the agent under whose context *span* executed.

        Walks ``parent_id`` upward until it finds a span with no parent.
        That root span's ``tool_name`` (Langfuse "name") is the agent id;
        we strip a sentinel like ``"agent-"`` if present. Falls back to
        the trace_id when no root is reachable.
        """
        # 1. Explicit metadata override always wins.
        explicit = self._meta_get(span, "agent_id")
        if explicit:
            return str(explicit)

        # 2. Walk up to the root.
        cur: Optional[ParsedSpan] = span
        visited: set[str] = set()
        while cur is not None and cur.parent_id and cur.parent_id not in visited:
            visited.add(cur.id)
            parent = span_index.get(cur.parent_id)
            if parent is None:
                break
            cur = parent

        if cur is not None and cur.parent_id is None and cur.tool_name:
            return _normalise_agent_id(cur.tool_name)

        # 3. Trace-level fallback.
        return f"agent::{span.trace_id}"

    def build_agent_node(self, span: ParsedSpan, agent_id: str) -> AgentNode:
        """Construct the AgentNode the first time we encounter *agent_id*.

        The seeding span (typically the root) supplies model/role metadata.
        """
        role = self._meta_get(span, "role") or self._meta_get(span, "agent_role")
        return AgentNode(
            id=agent_id,
            role=str(role) if role else "agent",
            system_prompt=self._meta_get(span, "system_prompt") or "",
            declared_capabilities=list(self._meta_get(span, "capabilities") or []),
            metadata={
                "trace_id": span.trace_id,
                "model": span.model,
                "first_seen_span_id": span.id,
            },
        )

    # -- tool resolution ---------------------------------------------------

    def resolve_tool_id(self, span: ParsedSpan) -> Optional[str]:
        """Return a tool id if *span* represents a tool invocation, else None."""
        explicit = (
            self._meta_get(span, "tool_id")
            or self._meta_get(span, "tool")
        )
        if explicit:
            return str(explicit)

        if (span.span_type or "").upper() == "TOOL":
            return _normalise_tool_id(span.tool_name or span.id)

        name = (span.tool_name or "").strip().lower()
        if name.startswith(("tool-", "tool:", "tool/")):
            return _normalise_tool_id(name)

        return None

    def build_tool_node(self, span: ParsedSpan, tool_id: str) -> ToolNode:
        return ToolNode(
            id=tool_id,
            name=span.tool_name or tool_id,
            description=str(self._meta_get(span, "description") or ""),
            metadata={
                "first_seen_span_id": span.id,
                "first_seen_trace_id": span.trace_id,
            },
        )

    # -- edge classification ----------------------------------------------

    def infer_edge_type(self, span: ParsedSpan, is_tool_call: bool) -> EdgeType:
        if is_tool_call:
            return EdgeType.TOOL_INVOCATION
        if (span.span_type or "").upper() == "GENERATION":
            return EdgeType.REASONING
        return EdgeType.UNKNOWN

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _meta_get(span: ParsedSpan, key: str) -> Any:
        meta = span.metadata
        if isinstance(meta, dict):
            return meta.get(key)
        return None


def _normalise_agent_id(raw: str) -> str:
    """Strip optional ``agent-``/``agent:`` prefixes; otherwise pass through."""
    s = raw.strip()
    for prefix in ("agent-", "agent:", "agent/"):
        if s.lower().startswith(prefix):
            return s[len(prefix):]
    return s


def _normalise_tool_id(raw: str) -> str:
    """Strip ``tool-``/``tool:``/``tool/`` prefixes; lowercase; trim."""
    s = raw.strip()
    for prefix in ("tool-", "tool:", "tool/"):
        if s.lower().startswith(prefix):
            return s[len(prefix):]
    return s


# ---------------------------------------------------------------------------
# Event Monitor — public API
# ---------------------------------------------------------------------------

class EventMonitor:
    """Streaming ingestor that builds an ``InteractionGraph`` (kind="dynamic").

    Parameters
    ----------
    resolver:
        Strategy object for span → (agent, tool, edge_type) resolution. Inject
        a custom ``SpanResolver`` subclass to adapt to a non-standard MAS
        tracing scheme. Defaults to ``SpanResolver()``.
    system_state_provider:
        Optional callable invoked per edge to attach a snapshot of current
        system state (active tools, memory contents, etc.) to the edge
        metadata. Receives the ParsedSpan being ingested; must return a
        ``dict`` (or empty dict).
    graph:
        Inject a pre-existing graph to append into. Defaults to a new empty
        dynamic graph.

    Notes
    -----
    Synchronous, single-threaded. The SentinelAgent paper calls for buffered
    non-blocking ingest "to avoid slowing down the main execution path"; that
    transport is left to the caller — wrap ``ingest_span`` in a queue, asyncio
    task, or background worker as needed for your deployment.
    """

    def __init__(
        self,
        *,
        resolver: Optional[SpanResolver] = None,
        system_state_provider: Optional[SystemStateProvider] = None,
        graph: Optional[InteractionGraph] = None,
    ) -> None:
        self._resolver = resolver or SpanResolver()
        self._state_provider = system_state_provider
        self.graph: InteractionGraph = graph or InteractionGraph(kind="dynamic")
        self._span_index: dict[str, ParsedSpan] = {}

    # -- public ------------------------------------------------------------

    def ingest_span(self, span: ParsedSpan) -> Optional[Edge]:
        """Process one span; mutate the graph; return the new edge (or None).

        Returns ``None`` for the root span of a trace (there's no upstream
        actor to form an edge from — the root just registers the agent node).
        """
        if not isinstance(span, ParsedSpan):
            raise TypeError(
                f"span must be ParsedSpan, got {type(span).__name__}"
            )
        if not span.id:
            raise ValueError("ParsedSpan must have a non-empty id")

        self._span_index[span.id] = span

        # 1. Agent context for this span.
        agent_id = self._resolver.resolve_agent_id(span, self._span_index)
        if agent_id not in self.graph.nodes:
            self.graph.add_node(self._resolver.build_agent_node(span, agent_id))

        # 2. Tool detection.
        tool_id = self._resolver.resolve_tool_id(span)
        if tool_id and tool_id not in self.graph.nodes:
            self.graph.add_node(self._resolver.build_tool_node(span, tool_id))

        # 3. Root spans register the agent but emit no edge — there is no
        #    upstream actor for an edge to flow from.
        if not span.parent_id:
            log.debug(
                "ingest_span: %s is a root span; registered agent %s, no edge",
                span.id, agent_id,
            )
            return None

        # 4. Build the typed edge.
        edge_type = self._resolver.infer_edge_type(span, tool_id is not None)
        target = tool_id if tool_id else agent_id  # REASONING ⇒ self-loop

        edge = Edge(
            id=span.id,                       # spans map 1:1 to edges
            source=agent_id,
            target=target,
            edge_type=edge_type,
            timestamp=span.timestamp or span.start_time or _safe_now(),
            message_content=span.output_text or None,
            tool_args=self._extract_tool_args(span) if tool_id else None,
            response_payload=span.output_text if tool_id else None,
            agent_thoughts=span.input_text if edge_type is EdgeType.REASONING else None,
            span_id=span.id,
            trace_id=span.trace_id,
            parent_edge_id=span.parent_id,
            metadata={
                "span_type": span.span_type,
                "latency": span.latency,
                "total_tokens": span.total_tokens,
                "cost": span.cost,
                "model": span.model,
            },
        )

        # 5. Optional runtime-state augmentation.
        if self._state_provider is not None:
            try:
                state = self._state_provider(span) or {}
                if state:
                    edge.metadata["system_state"] = state
            except Exception:
                # State providers must not break ingest.
                log.exception(
                    "system_state_provider raised on span %s — ignoring", span.id
                )

        return self.graph.add_edge(edge)

    def ingest_many(self, spans: Iterable[ParsedSpan]) -> InteractionGraph:
        """Ingest an iterable of spans. Returns the (now-updated) graph."""
        for span in spans:
            self.ingest_span(span)
        return self.graph

    # -- diagnostics -------------------------------------------------------

    def summary(self) -> dict:
        return self.graph.summary()

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _extract_tool_args(span: ParsedSpan) -> Optional[dict[str, Any]]:
        """Best-effort: surface tool args from input_text or metadata."""
        meta = span.metadata if isinstance(span.metadata, dict) else {}
        args = meta.get("tool_args") or meta.get("arguments") or meta.get("args")
        if isinstance(args, dict):
            return args
        if span.input_text:
            return {"raw_input": span.input_text}
        return None


# ---------------------------------------------------------------------------
# Convenience: one-shot batch build
# ---------------------------------------------------------------------------

def build_dynamic_graph(
    spans: Iterable[ParsedSpan],
    *,
    resolver: Optional[SpanResolver] = None,
    system_state_provider: Optional[SystemStateProvider] = None,
) -> InteractionGraph:
    """Convenience: build a dynamic graph from a complete span list."""
    mon = EventMonitor(
        resolver=resolver, system_state_provider=system_state_provider
    )
    mon.ingest_many(spans)
    return mon.graph


def _safe_now():
    """Fallback timestamp when the span carries none — keeps Edge invariant happy."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
