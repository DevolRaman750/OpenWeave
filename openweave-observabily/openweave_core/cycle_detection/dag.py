"""
dag.py — Weighted DAG construction from a trace trajectory + sibling-node
         extraction for flagged cycle candidates.

Pipeline position
-----------------
CDCS (flagged CycleCandidates)
  -> build_dag_and_siblings(trajectory, flagged_spans)
  -> (TraceDAG, list[SiblingGroup])
  -> semantic similarity stage

Graph definition
----------------
  Nodes  S  : every span in trajectory T
  Edges  E  : si -> sj  iff  parent_id(sj) == id(si)
  Weight w  : times the (si, sj) traversal appears in T

Siblings of span s: all children of parent_id(s) in Gt, i.e.
  siblings(s) = { x in S | parent_id(x) == parent_id(s) }
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from openweave_core.models.span import ParsedSpan

SpanLike = Union[ParsedSpan, dict[str, Any]]


# ---------------------------------------------------------------------------
# Span field accessors (normalise ParsedSpan <-> dict)
# ---------------------------------------------------------------------------

def _sid(span: SpanLike) -> str:
    """Return span_id."""
    if isinstance(span, ParsedSpan):
        return span.id
    return span.get("id") or span.get("span_id") or ""


def _pid(span: SpanLike) -> Optional[str]:
    """Return parent_id, or None for root spans."""
    if isinstance(span, ParsedSpan):
        return span.parent_id
    return span.get("parent_id") or span.get("parent_span_id") or None


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class TraceDAG:
    """Weighted directed acyclic graph for one trace.

    Attributes
    ----------
    nodes        : span_id -> span object
    children     : parent_id -> list of unique child span_ids
    edge_weights : (parent_id, child_id) -> traversal count
    roots        : span_ids with no parent (depth-0 entry points)
    """

    nodes: dict[str, SpanLike]
    children: dict[str, list[str]]
    edge_weights: dict[tuple[str, str], int]
    roots: list[str]


@dataclass
class SiblingGroup:
    """All children of a shared parent, surfaced by a flagged span.

    Attributes
    ----------
    parent_id        : the common parent span_id
    flagged_span_ids : spans inside the flagged cycle that have this parent
    siblings         : complete set of children of parent_id in Gt
                       (includes flagged spans themselves)
    """

    parent_id: str
    flagged_span_ids: list[str] = field(default_factory=list)
    siblings: list[SpanLike] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — Build the full weighted DAG
# ---------------------------------------------------------------------------

def build_trace_dag(trajectory: list[SpanLike]) -> TraceDAG:
    """Construct Gt from the complete trajectory T.

    Parameters
    ----------
    trajectory:
        All spans for a single trace_id (order does not matter here;
        the DAG is topology-based, not time-based).

    Returns
    -------
    TraceDAG
        Adjacency lists + edge-weight Counter for the full trace graph.

    Raises
    ------
    ValueError
        If any span has an empty id.
    """
    nodes: dict[str, SpanLike] = {}
    _children_set: dict[str, set[str]] = defaultdict(set)
    edge_counter: Counter[tuple[str, str]] = Counter()
    roots: list[str] = []

    for span in trajectory:
        sid = _sid(span)
        if not sid:
            raise ValueError(f"Span missing id field: {span!r}")

        nodes[sid] = span
        pid = _pid(span)

        if pid:
            _children_set[pid].add(sid)
            edge_counter[(pid, sid)] += 1  # weight: traversal frequency
        else:
            roots.append(sid)

    children: dict[str, list[str]] = {
        pid: sorted(cids) for pid, cids in _children_set.items()
    }

    return TraceDAG(
        nodes=nodes,
        children=children,
        edge_weights=dict(edge_counter),
        roots=roots,
    )


# ---------------------------------------------------------------------------
# Step 2 — Find sibling groups for flagged spans
# ---------------------------------------------------------------------------

def find_sibling_groups(
    dag: TraceDAG,
    flagged_spans: list[SpanLike],
) -> list[SiblingGroup]:
    """Locate sibling sets for each unique parent found in *flagged_spans*.

    A sibling set for span s = all children of parent_id(s) in Gt.
    One SiblingGroup is emitted per unique parent_id; if multiple flagged
    spans share the same parent they are merged into a single group.

    Parameters
    ----------
    dag:
        Full TraceDAG (output of ``build_trace_dag``).
    flagged_spans:
        Flat list of spans extracted from CycleCandidate.first_occurrence.

    Returns
    -------
    list[SiblingGroup]
        One entry per unique parent_id encountered in *flagged_spans*.
        Groups are ordered by number of flagged spans per parent (desc).
        Root spans (no parent) are skipped.
    """
    parent_to_flagged: dict[str, set[str]] = defaultdict(set)

    for span in flagged_spans:
        pid = _pid(span)
        if pid is None:
            continue
        parent_to_flagged[pid].add(_sid(span))

    groups: list[SiblingGroup] = []

    for pid, flagged_ids in parent_to_flagged.items():
        child_ids = dag.children.get(pid, [])
        sibling_spans = [dag.nodes[cid] for cid in child_ids if cid in dag.nodes]

        groups.append(
            SiblingGroup(
                parent_id=pid,
                flagged_span_ids=sorted(flagged_ids),
                siblings=sibling_spans,
            )
        )

    groups.sort(key=lambda g: len(g.flagged_span_ids), reverse=True)
    return groups


# ---------------------------------------------------------------------------
# Convenience: single-call entry point
# ---------------------------------------------------------------------------

def build_dag_and_siblings(
    trajectory: list[SpanLike],
    flagged_spans: list[SpanLike],
) -> tuple[TraceDAG, list[SiblingGroup]]:
    """Build Gt from *trajectory* and extract sibling groups for *flagged_spans*.

    Parameters
    ----------
    trajectory:
        Complete span list for the trace (all spans, not just flagged ones).
    flagged_spans:
        Spans from CDCS-flagged CycleCandidates (e.g. candidate.first_occurrence).

    Returns
    -------
    (TraceDAG, list[SiblingGroup])
        DAG for downstream graph queries + sibling groups ready for semantic
        similarity confirmation.

    Raises
    ------
    TypeError
        If either argument is not a list.
    ValueError
        If any span in *trajectory* has an empty id.
    """
    if not isinstance(trajectory, list):
        raise TypeError(f"trajectory must be list, got {type(trajectory).__name__}")
    if not isinstance(flagged_spans, list):
        raise TypeError(f"flagged_spans must be list, got {type(flagged_spans).__name__}")

    dag = build_trace_dag(trajectory)
    groups = find_sibling_groups(dag, flagged_spans)
    return dag, groups
