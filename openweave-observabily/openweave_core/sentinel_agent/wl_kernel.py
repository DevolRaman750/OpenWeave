"""
wl_kernel.py — Weisfeiler-Lehman subtree kernel + Edge-Distance kernel.

The Attack-Path Matcher (Module 2) uses these to score *structural*
similarity between a live dynamic graph and a known attack-pattern graph.

Two kernels are implemented:

  1. WL subtree kernel
     -----------------
     Iteratively relabels each node by hashing its own label together with the
     sorted multiset of its directed-neighbour labels. After ``h`` iterations
     (default 3, captures up to 3-hop branch points), each node's label
     encodes a hash of its h-hop subtree. The kernel score is the inner
     product of label-frequency vectors taken over every iteration —
     equivalent to counting shared subtree patterns between two graphs.

  2. Edge-Distance kernel
     --------------------
     Complements WL by scoring *edge co-occurrence weighted by temporal
     proximity*. Two edges with the same (src_type, tgt_type, edge_type)
     label contribute  exp(-|Δt|/τ)  to the score, where Δt is the time
     gap between the edges and τ is the decay constant. This injects the
     "edge weights = temporal execution timings" requirement from the spec.

Both kernels are normalised against self-kernels so the combined score lives
in [0, 1] (1 = perfect match). The matcher exposes a blended score:

    combined = α · K̂_WL + (1 - α) · K̂_edge_dist
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    EdgeType,
    Node,
    ToolNode,
)


DEFAULT_DEPTH: int = 3
DEFAULT_TIME_DECAY_SECONDS: float = 10.0
DEFAULT_BLEND: float = 0.5  # 0.5 WL + 0.5 edge-distance


# ---------------------------------------------------------------------------
# Initial labels
# ---------------------------------------------------------------------------

def _initial_node_label(node: Node) -> str:
    """Coarse seed label — node type only.

    Why so coarse: the WL kernel must match topology (the *shape* of an
    interaction), not identity. A redundant-tool-cycle pattern is the same
    structural exploit whether the tool is ``search`` or ``vector_lookup``.
    Fine-grained identity matching is handled by the semantic correlator
    downstream — pattern + live graph compare on role/name/keywords there.
    """
    if isinstance(node, AgentNode):
        return "A"
    if isinstance(node, ToolNode):
        return "T"
    return "N"


def _node_kind_label(node: Node) -> str:
    """Coarser label used when building edge-type triples."""
    if isinstance(node, AgentNode):
        return "agent"
    if isinstance(node, ToolNode):
        return "tool"
    return "?"


def _hash(text: str) -> str:
    """Stable short label hash — collisions are astronomically unlikely at 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Weisfeiler-Lehman relabelling
# ---------------------------------------------------------------------------

def wl_relabel(
    graph: InteractionGraph, depth: int = DEFAULT_DEPTH
) -> list[dict[str, str]]:
    """Return ``[labels_0, labels_1, …, labels_depth]``.

    ``labels_i[node_id]`` is the WL label of *node_id* after ``i`` rounds.
    A "round" combines a node's current label with its (sorted, typed)
    neighbour labels and re-hashes. Directed edges contribute distinct
    "out" / "in" markers so direction is preserved in the signature.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")

    labels: dict[str, str] = {
        nid: _initial_node_label(node) for nid, node in graph.nodes.items()
    }
    history: list[dict[str, str]] = [dict(labels)]

    for _ in range(depth):
        new_labels: dict[str, str] = {}
        for nid in graph.nodes:
            neigh: list[tuple[str, str, str]] = []
            # Skip self-loops (e.g. agent REASONING edges) — they encode
            # intra-node thinking, not relational structure, and would
            # otherwise drown out pattern matches against simpler exemplars.
            for e in graph.outgoing_edges(nid):
                if e.source == e.target:
                    continue
                neigh.append(("out", e.edge_type.value, labels.get(e.target, "")))
            for e in graph.incoming_edges(nid):
                if e.source == e.target:
                    continue
                neigh.append(("in", e.edge_type.value, labels.get(e.source, "")))
            neigh.sort()
            signature = f"{labels[nid]}|" + "|".join(
                f"{d}:{t}:{l}" for d, t, l in neigh
            )
            new_labels[nid] = _hash(signature)
        labels = new_labels
        history.append(dict(labels))

    return history


# ---------------------------------------------------------------------------
# WL subtree kernel — inner product of label-frequency vectors
# ---------------------------------------------------------------------------

def wl_label_counters(
    graph: InteractionGraph, depth: int = DEFAULT_DEPTH
) -> list[Counter]:
    """Per-iteration label-frequency Counters for *graph*.

    Precompute this once and reuse it across many kernel evaluations: the WL
    kernel between two graphs is just the inner product of their counter lists,
    so the matcher computes the live graph's counters a single time instead of
    re-relabelling it for every attack pattern in the library.
    """
    return [Counter(labels.values()) for labels in wl_relabel(graph, depth=depth)]


def wl_kernel_from_counters(
    counters1: list[Counter], counters2: list[Counter]
) -> float:
    """WL kernel from precomputed counters: Σ_iter Σ_label freq1·freq2."""
    score = 0.0
    for c1, c2 in zip(counters1, counters2):
        # Iterate the smaller multiset; labels absent from the other contribute 0.
        if len(c2) < len(c1):
            c1, c2 = c2, c1
        for label, n in c1.items():
            other = c2.get(label)
            if other:
                score += n * other
    return float(score)


def wl_subtree_kernel(
    g1: InteractionGraph,
    g2: InteractionGraph,
    depth: int = DEFAULT_DEPTH,
) -> float:
    """Sum over iterations of  Σ_label  freq1[label] · freq2[label]."""
    return wl_kernel_from_counters(
        wl_label_counters(g1, depth=depth),
        wl_label_counters(g2, depth=depth),
    )


def normalised_wl_kernel(
    g1: InteractionGraph,
    g2: InteractionGraph,
    depth: int = DEFAULT_DEPTH,
) -> float:
    """K̂(G₁, G₂) = K(G₁, G₂) / √(K(G₁, G₁) · K(G₂, G₂))  ∈ [0, 1]."""
    k12 = wl_subtree_kernel(g1, g2, depth=depth)
    k11 = wl_subtree_kernel(g1, g1, depth=depth)
    k22 = wl_subtree_kernel(g2, g2, depth=depth)
    denom = math.sqrt(k11 * k22)
    if denom <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, k12 / denom)))


# ---------------------------------------------------------------------------
# Edge-Distance kernel
# ---------------------------------------------------------------------------

def _edge_label(edge, graph: InteractionGraph) -> tuple[str, str, str]:
    src = graph.get_node(edge.source)
    tgt = graph.get_node(edge.target)
    return (
        _node_kind_label(src) if src else "?",
        _node_kind_label(tgt) if tgt else "?",
        edge.edge_type.value,
    )


def edge_label_buckets(graph: InteractionGraph) -> dict[tuple[str, str, str], list]:
    """Map (src_kind, tgt_kind, edge_type) → chronologically-sorted timestamps.

    Built once per graph so the matcher reuses the *live* graph's buckets for
    every library pattern instead of rescanning its whole edge set per pattern.
    Sorting enables the O(k log k) self-kernel recurrence below.
    """
    buckets: dict[tuple[str, str, str], list] = defaultdict(list)
    for e in graph.edges.values():
        buckets[_edge_label(e, graph)].append(e.timestamp)
    for ts in buckets.values():
        ts.sort()
    return buckets


def edge_self_kernel_from_buckets(
    buckets: dict[tuple[str, str, str], list],
    *,
    time_decay_seconds: float = DEFAULT_TIME_DECAY_SECONDS,
) -> float:
    """Exact self edge-distance kernel K(G, G) in O(E log E), not O(E²).

    The brute-force kernel sums exp(-|Δt|/τ) over *every ordered pair* of
    co-labelled edges. For one same-label bucket of ``k`` chronologically
    sorted timestamps that double sum is the diagonal (``k`` ones, Δt=0) plus
    twice the strictly-upper-triangle sum ``Σ_{i<j} exp(-(t_j - t_i)/τ)``.
    Because the timestamps are sorted, that upper sum telescopes:

        R_j = exp(-(t_j - t_{j-1})/τ) · (R_{j-1} + 1),   R_0 = 0
        Σ_{i<j} exp(-(t_j - t_i)/τ) = Σ_j R_j

    a numerically-stable O(k) recurrence — every factor is ≤ 1 so there is no
    exp-overflow — that yields the same value as the brute-force pair loop.
    This is what removes the wide/flat-trace cliff (one bucket of 3000 edges
    went from ~9M pair evaluations to ~3000).
    """
    if time_decay_seconds <= 0.0:
        raise ValueError(f"time_decay_seconds must be > 0, got {time_decay_seconds}")

    score = 0.0
    for ts in buckets.values():
        score += len(ts)  # diagonal: each edge vs itself, exp(0) = 1
        running = 0.0
        prev = None
        for t in ts:
            if prev is not None:
                gap = (t - prev).total_seconds()
                running = math.exp(-gap / time_decay_seconds) * (running + 1.0)
            score += 2.0 * running  # symmetric off-diagonal pairs (i<j and j>i)
            prev = t
    return float(score)


def edge_cross_kernel_from_buckets(
    live_buckets: dict[tuple[str, str, str], list],
    pattern: InteractionGraph,
    *,
    time_decay_seconds: float = DEFAULT_TIME_DECAY_SECONDS,
) -> float:
    """Cross edge-distance kernel K(live, pattern) reusing prebuilt live buckets.

    Mirrors ``edge_distance_kernel(live, pattern)`` exactly: for every pattern
    edge, sum exp(-|Δt|/τ) over the live edges that share its
    (src_kind, tgt_kind, edge_type) label. Pattern graphs are tiny, so this
    stays linear in the matched live bucket and is computed per pattern.
    """
    if time_decay_seconds <= 0.0:
        raise ValueError(f"time_decay_seconds must be > 0, got {time_decay_seconds}")

    score = 0.0
    for e2 in pattern.edges.values():
        bucket = live_buckets.get(_edge_label(e2, pattern))
        if not bucket:
            continue
        t2 = e2.timestamp
        for t1 in bucket:
            dt = abs((t1 - t2).total_seconds())
            score += math.exp(-dt / time_decay_seconds)
    return float(score)


def edge_distance_kernel(
    g1: InteractionGraph,
    g2: InteractionGraph,
    *,
    time_decay_seconds: float = DEFAULT_TIME_DECAY_SECONDS,
) -> float:
    """Σ over co-labelled edge pairs of  exp(-|Δt| / τ).

    Co-labelling is by (src_kind, tgt_kind, edge_type) — coarse enough that
    a single agent->tool->agent fan-out shows up consistently, fine enough to
    not collide unrelated topologies.

    Pattern graphs whose edges carry a sentinel timestamp (datetime.min) will
    naturally show high edge-distance similarity to *any* recent dynamic edge
    because exp(-huge/τ) → 0; in that case the kernel score collapses toward
    counting label-only matches with vanishing weights. To make pattern
    graphs comparable to live timestamps, the matcher normalises pattern-time
    relative to the live trace before calling this kernel.
    """
    return edge_cross_kernel_from_buckets(
        edge_label_buckets(g1), g2, time_decay_seconds=time_decay_seconds,
    )


def normalised_edge_distance_kernel(
    g1: InteractionGraph,
    g2: InteractionGraph,
    *,
    time_decay_seconds: float = DEFAULT_TIME_DECAY_SECONDS,
) -> float:
    k12 = edge_distance_kernel(g1, g2, time_decay_seconds=time_decay_seconds)
    k11 = edge_distance_kernel(g1, g1, time_decay_seconds=time_decay_seconds)
    k22 = edge_distance_kernel(g2, g2, time_decay_seconds=time_decay_seconds)
    denom = math.sqrt(k11 * k22)
    if denom <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, k12 / denom)))


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

@dataclass
class StructuralScore:
    """Decomposed similarity report for diagnostics."""
    combined: float        # blended final score in [0, 1]
    wl: float              # normalised WL component
    edge_distance: float   # normalised edge-distance component
    blend: float           # α used in the linear combination


def structural_similarity(
    live: InteractionGraph,
    pattern: InteractionGraph,
    *,
    depth: int = DEFAULT_DEPTH,
    time_decay_seconds: float = DEFAULT_TIME_DECAY_SECONDS,
    blend: float = DEFAULT_BLEND,
) -> StructuralScore:
    """Compute the blended WL × edge-distance similarity.

    Parameters
    ----------
    live, pattern :
        Two InteractionGraphs. ``pattern`` is the attack-pattern exemplar.
    depth :
        Number of WL refinement iterations (≤ 3 captures the 3-hop subtree
        the spec calls for).
    time_decay_seconds :
        τ in exp(-|Δt|/τ). Larger τ ⇒ temporal mismatches matter less.
    blend :
        α in  combined = α · K̂_WL + (1-α) · K̂_edge_dist.

    Returns
    -------
    StructuralScore with all three values for diagnostics. The matcher
    typically thresholds on ``StructuralScore.combined``.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend}")

    wl = normalised_wl_kernel(live, pattern, depth=depth)
    ed = normalised_edge_distance_kernel(
        live, pattern, time_decay_seconds=time_decay_seconds
    )
    combined = blend * wl + (1.0 - blend) * ed
    return StructuralScore(
        combined=float(max(0.0, min(1.0, combined))),
        wl=wl,
        edge_distance=ed,
        blend=blend,
    )
