"""
path_matcher.py — Module 2: structural + semantic correlation + double-check.

Pipeline per attack-path entry
------------------------------
    1. structural pass   : WL × Edge-Distance kernel  →  StructuralScore
    2. semantic pass     : check live subgraph against AttackPath.signature
    3. emit Finding      : if both pass, attach matched node/edge ids
    4. (handled in BehaviorAnalyzer): if path matched but no local findings,
       trigger double-check by re-running judges with a stricter threshold

Why two passes
--------------
The kernel alone is fast but topology-blind to *content*. The semantic
correlation step ensures we don't flag every legitimate retry loop as a
"redundant cycle" — the signature requires actual repetition counts /
text markers / authorisation violations to align with the named exploit.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from openweave_core.sentinel_agent.attack_paths import (
    AttackPath,
    AttackPathLibrary,
    SemanticSignature,
)
from openweave_core.sentinel_agent.findings import (
    Finding,
    FindingCategory,
    FindingSource,
    Severity,
)
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import Edge, EdgeType
from openweave_core.sentinel_agent.wl_kernel import (
    DEFAULT_BLEND,
    DEFAULT_DEPTH,
    DEFAULT_TIME_DECAY_SECONDS,
    StructuralScore,
    edge_cross_kernel_from_buckets,
    edge_label_buckets,
    edge_self_kernel_from_buckets,
    wl_kernel_from_counters,
    wl_label_counters,
)


DEFAULT_STRUCTURAL_THRESHOLD: float = 0.5
DEFAULT_SEMANTIC_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Output of one match attempt
# ---------------------------------------------------------------------------

@dataclass
class PathMatch:
    """Diagnostic record for a single (live, attack_path) comparison."""
    path: AttackPath
    structural: StructuralScore
    semantic_score: float
    semantic_evidence: dict = field(default_factory=dict)
    matched: bool = False
    finding: Optional[Finding] = None
    # Live-graph subjects that contributed to the match — feeds root-cause:
    origin_node_ids: list[str] = field(default_factory=list)
    matched_edge_ids: list[str] = field(default_factory=list)


@dataclass
class _PatternKernels:
    """Per-pattern self-kernel terms that are invariant across live traces.

    The WL kernel ignores timestamps and the edge-distance *self*-kernel depends
    only on intra-pattern time gaps (so timestamp rebasing, a constant shift,
    leaves it unchanged). Both are therefore independent of the live graph and
    cached once per attack path instead of recomputed on every ``search``.
    """
    wl_counters: list[Counter]
    wl_self: float
    edge_self: float


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class AttackPathMatcher:
    """Continuously compare a live dynamic graph against an attack library."""

    def __init__(
        self,
        *,
        library: AttackPathLibrary,
        structural_threshold: float = DEFAULT_STRUCTURAL_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        kernel_depth: int = DEFAULT_DEPTH,
        time_decay_seconds: float = DEFAULT_TIME_DECAY_SECONDS,
        kernel_blend: float = DEFAULT_BLEND,
    ) -> None:
        self._library = library
        self._struct_thr = structural_threshold
        self._sem_thr = semantic_threshold
        self._depth = kernel_depth
        self._tau = time_decay_seconds
        self._blend = kernel_blend
        # Pattern self-kernels are live-graph-independent — memoise per path id.
        self._pattern_cache: dict[str, _PatternKernels] = {}

    @property
    def library(self) -> AttackPathLibrary:
        return self._library

    # -----------------------------------------------------------------
    # Main entry — score every path in the library against the live graph
    # -----------------------------------------------------------------

    def search(
        self,
        live_graph: InteractionGraph,
        *,
        require_match: bool = True,
    ) -> list[PathMatch]:
        """Score every library path. Returns the per-path PathMatch records.

        ``require_match=True`` filters the result to only matched paths.
        ``require_match=False`` returns all scoring records (useful for
        debugging or for the BehaviorAnalyzer to decide when to skip the
        double-check entirely).
        """
        # Normalise pattern timestamps to the live window so the temporal
        # term in the edge-distance kernel produces meaningful weights.
        live_anchor = _earliest_timestamp(live_graph)

        # Precompute the live graph's self-kernels ONCE. Previously each library
        # pattern triggered a fresh structural_similarity(), which recomputed the
        # O(E²) edge-distance self-kernel K(live, live) from scratch — the cause
        # of the multi-minute sentinel cliff on wide/flat traces.
        live_wl_counters = wl_label_counters(live_graph, depth=self._depth)
        live_wl_self = wl_kernel_from_counters(live_wl_counters, live_wl_counters)
        live_buckets = edge_label_buckets(live_graph)
        live_edge_self = edge_self_kernel_from_buckets(
            live_buckets, time_decay_seconds=self._tau
        )

        results: list[PathMatch] = []
        for path in self._library:
            structural = self._structural_score(
                path,
                live_anchor=live_anchor,
                live_wl_counters=live_wl_counters,
                live_wl_self=live_wl_self,
                live_buckets=live_buckets,
                live_edge_self=live_edge_self,
            )

            if structural.combined < self._struct_thr:
                results.append(PathMatch(
                    path=path,
                    structural=structural,
                    semantic_score=0.0,
                    matched=False,
                ))
                continue

            sem_score, sem_evidence, contributing = _semantic_correlate(
                live_graph, path
            )
            matched = sem_score >= self._sem_thr
            finding = (
                self._build_finding(path, structural, sem_score, sem_evidence,
                                    contributing)
                if matched else None
            )
            results.append(PathMatch(
                path=path,
                structural=structural,
                semantic_score=sem_score,
                semantic_evidence=sem_evidence,
                matched=matched,
                finding=finding,
                origin_node_ids=contributing["origin_nodes"],
                matched_edge_ids=contributing["edge_ids"],
            ))

        if require_match:
            return [r for r in results if r.matched]
        return results

    # -----------------------------------------------------------------
    # Structural scoring (precomputed live self-kernels + cached pattern self)
    # -----------------------------------------------------------------

    def _pattern_kernels(self, path: AttackPath) -> _PatternKernels:
        """Self-kernels for *path*, memoised (independent of the live graph)."""
        cached = self._pattern_cache.get(path.id)
        if cached is not None:
            return cached
        pattern = path.pattern_graph
        wl_counters = wl_label_counters(pattern, depth=self._depth)
        wl_self = wl_kernel_from_counters(wl_counters, wl_counters)
        # Edge self-kernel is shift-invariant, so the un-rebased pattern is fine.
        edge_self = edge_self_kernel_from_buckets(
            edge_label_buckets(pattern), time_decay_seconds=self._tau
        )
        kern = _PatternKernels(
            wl_counters=wl_counters, wl_self=wl_self, edge_self=edge_self
        )
        self._pattern_cache[path.id] = kern
        return kern

    def _structural_score(
        self,
        path: AttackPath,
        *,
        live_anchor,
        live_wl_counters: list[Counter],
        live_wl_self: float,
        live_buckets: dict,
        live_edge_self: float,
    ) -> StructuralScore:
        """Blended WL × edge-distance score, numerically identical to
        ``structural_similarity(live, rebased_pattern)`` but reusing the
        precomputed live self-kernels and cached pattern self-kernels.
        """
        pk = self._pattern_kernels(path)

        # WL is timestamp-independent, so the cached (un-rebased) pattern
        # counters give the exact same cross term.
        k12_wl = wl_kernel_from_counters(live_wl_counters, pk.wl_counters)
        denom_wl = math.sqrt(live_wl_self * pk.wl_self)
        wl = 0.0 if denom_wl <= 0.0 else max(0.0, min(1.0, k12_wl / denom_wl))

        # The edge-distance *cross* term depends on absolute time gaps, so the
        # pattern must be rebased into the live window first (self terms don't).
        normed_pattern = _rebase_pattern_timestamps(path.pattern_graph, live_anchor)
        k12_ed = edge_cross_kernel_from_buckets(
            live_buckets, normed_pattern, time_decay_seconds=self._tau
        )
        denom_ed = math.sqrt(live_edge_self * pk.edge_self)
        ed = 0.0 if denom_ed <= 0.0 else max(0.0, min(1.0, k12_ed / denom_ed))

        combined = self._blend * wl + (1.0 - self._blend) * ed
        return StructuralScore(
            combined=float(max(0.0, min(1.0, combined))),
            wl=wl,
            edge_distance=ed,
            blend=self._blend,
        )

    # -----------------------------------------------------------------
    # Finding constructor
    # -----------------------------------------------------------------

    def _build_finding(
        self,
        path: AttackPath,
        structural: StructuralScore,
        sem_score: float,
        sem_evidence: dict,
        contributing: dict,
    ) -> Finding:
        confidence = min(1.0, 0.5 * structural.combined + 0.5 * sem_score)
        return Finding(
            id=f"path::{path.id}",
            severity=path.severity,
            category=FindingCategory.ATTACK_PATH,
            code=f"ATTACK_PATH_{path.id.upper()}",
            message=(
                f"Live execution matched attack pattern '{path.name}' "
                f"(structural={structural.combined:.2f}, semantic={sem_score:.2f})."
            ),
            subject_id=path.id,
            confidence=confidence,
            source=FindingSource.PATH_MATCHER,
            remediation=path.remediation,
            evidence={
                "path_id": path.id,
                "path_name": path.name,
                "description": path.description,
                "structural_score": structural.combined,
                "wl_score": structural.wl,
                "edge_distance_score": structural.edge_distance,
                "semantic_score": sem_score,
                "semantic_evidence": sem_evidence,
                "origin_nodes": contributing["origin_nodes"],
                "edge_ids": contributing["edge_ids"],
            },
        )


# ---------------------------------------------------------------------------
# Semantic correlation
# ---------------------------------------------------------------------------

def _semantic_correlate(
    live: InteractionGraph,
    path: AttackPath,
) -> tuple[float, dict, dict]:
    """Score live subgraph against the path's SemanticSignature.

    Returns ``(score, evidence, contributing)`` where:
      score        : in [0, 1] — fraction of signature conditions satisfied,
                     averaged across the active conditions.
      evidence     : dict of per-condition counts / matches.
      contributing : ``{"origin_nodes": [...], "edge_ids": [...]}`` of live
                     subjects that triggered the signature.
    """
    sig: SemanticSignature = path.signature
    conditions_active = 0
    conditions_satisfied = 0
    evidence: dict = {}
    origin_nodes: set[str] = set()
    edge_ids: set[str] = set()

    # 1) Repeated-edge condition
    if sig.min_repeats > 0 and sig.repeated_edge_type is not None:
        conditions_active += 1
        repeats_per_source: Counter[str] = Counter()
        edges_per_source: dict[str, list[str]] = {}
        for e in live.edges.values():
            if e.edge_type is sig.repeated_edge_type:
                repeats_per_source[e.source] += 1
                edges_per_source.setdefault(e.source, []).append(e.id)

        offenders = {
            src: count
            for src, count in repeats_per_source.items()
            if count >= sig.min_repeats
        }
        evidence["repeats_per_source"] = dict(repeats_per_source)
        evidence["repeat_offenders"] = offenders
        if offenders:
            conditions_satisfied += 1
            for src in offenders:
                origin_nodes.add(src)
                edge_ids.update(edges_per_source.get(src, ()))

    # 2) Text-keyword condition
    if sig.text_keywords:
        conditions_active += 1
        keyword_hits: list[tuple[str, str]] = []  # (edge_id, matched_keyword)
        lowered_keywords = tuple(k.lower() for k in sig.text_keywords)
        for e in live.edges.values():
            text = _aggregate_edge_text(e)
            if not text:
                continue
            lt = text.lower()
            for kw in lowered_keywords:
                if kw in lt:
                    keyword_hits.append((e.id, kw))
                    edge_ids.add(e.id)
                    origin_nodes.add(e.source)
                    break
        evidence["keyword_hits"] = keyword_hits
        if keyword_hits:
            conditions_satisfied += 1

    # 3) Unauthorised-relation condition (needs static graph injected by analyzer)
    if sig.requires_unauth:
        conditions_active += 1
        # We can't access the static graph here directly — the analyzer
        # cross-checks. We mark this condition as *satisfied* if any rule-engine
        # finding for unauthorised tool use exists on the live graph metadata.
        unauth_edges = [
            e for e in live.edges.values()
            if e.metadata.get("rule_unauthorised") is True
        ]
        evidence["unauth_edge_count"] = len(unauth_edges)
        if unauth_edges:
            conditions_satisfied += 1
            for e in unauth_edges:
                edge_ids.add(e.id)
                origin_nodes.add(e.source)

    # 4) Chain condition: A → A (message) followed by A → T (tool)
    if sig.requires_chain:
        conditions_active += 1
        chains: list[tuple[str, str]] = []
        for e_msg in live.edges.values():
            if e_msg.edge_type is not EdgeType.MESSAGE:
                continue
            for e_tool in live.outgoing_edges(e_msg.target):
                if (e_tool.edge_type is EdgeType.TOOL_INVOCATION
                        and e_tool.timestamp >= e_msg.timestamp):
                    chains.append((e_msg.id, e_tool.id))
                    origin_nodes.add(e_msg.source)
                    edge_ids.update([e_msg.id, e_tool.id])
        evidence["chains"] = chains
        if chains:
            conditions_satisfied += 1

    # If signature declares no conditions, fall back to structural-only match.
    if conditions_active == 0:
        return 1.0, {"note": "no semantic conditions; structural-only"}, {
            "origin_nodes": [], "edge_ids": [],
        }

    score = conditions_satisfied / conditions_active
    return float(score), evidence, {
        "origin_nodes": sorted(origin_nodes),
        "edge_ids": sorted(edge_ids),
    }


def _aggregate_edge_text(edge: Edge) -> str:
    parts: list[str] = []
    if edge.message_content:
        parts.append(edge.message_content)
    if edge.agent_thoughts:
        parts.append(edge.agent_thoughts)
    if edge.tool_args:
        parts.append(str(edge.tool_args))
    if edge.response_payload:
        parts.append(str(edge.response_payload))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _earliest_timestamp(graph: InteractionGraph):
    from datetime import datetime
    edges = list(graph.edges.values())
    if not edges:
        return datetime.utcnow()
    return min(e.timestamp for e in edges)


def _rebase_pattern_timestamps(
    pattern: InteractionGraph, live_anchor
) -> InteractionGraph:
    """Return a shallow copy of *pattern* with edge timestamps shifted to live_anchor.

    This makes the edge-distance kernel meaningful — without it, pattern
    timestamps from year 2000 would always be billions of seconds away from
    live timestamps, sending exp(-Δt/τ) to ~0 for every pair.
    """
    if not pattern.edges:
        return pattern
    earliest_pat = min(e.timestamp for e in pattern.edges.values())

    rebased = InteractionGraph(kind=pattern.kind)
    for nid, node in pattern.nodes.items():
        rebased.nodes[nid] = node
    for eid, edge in pattern.edges.items():
        new_edge = Edge(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            edge_type=edge.edge_type,
            timestamp=live_anchor + (edge.timestamp - earliest_pat),
            message_content=edge.message_content,
            tool_args=edge.tool_args,
            response_payload=edge.response_payload,
            agent_thoughts=edge.agent_thoughts,
            span_id=edge.span_id,
            trace_id=edge.trace_id,
            parent_edge_id=edge.parent_edge_id,
            metadata=dict(edge.metadata),
        )
        rebased.edges[eid] = new_edge
        rebased._outgoing[edge.source].append(eid)
        rebased._incoming[edge.target].append(eid)
    return rebased
