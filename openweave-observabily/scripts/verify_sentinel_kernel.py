"""Exactness gate for the sentinel kernel optimization.

Proves the optimized AttackPathMatcher (precomputed live self-kernels +
O(k log k) edge self-kernel + cached pattern self-kernels) produces structural
scores and match decisions that are NUMERICALLY IDENTICAL to a from-scratch
brute-force reference that re-implements the ORIGINAL O(E²) formulas inline.

Run from openweave-observabily/:  python scripts/verify_sentinel_kernel.py
"""
import math
import sys
from collections import Counter

sys.path.insert(0, ".")

from openweave_core.sentinel_agent import build_dynamic_graph, builtin_library
from openweave_core.sentinel_agent.path_matcher import (
    AttackPathMatcher,
    _earliest_timestamp,
    _rebase_pattern_timestamps,
)
from openweave_core.sentinel_agent.wl_kernel import (
    DEFAULT_BLEND,
    DEFAULT_DEPTH,
    DEFAULT_TIME_DECAY_SECONDS,
    _edge_label,
    wl_relabel,
)

from scripts.stress_test import (
    fx_redundant_cycle,
    fx_both,
    fx_many_agents,
    fx_big_flat,
    fx_wide_siblings,
    fx_deep,
)

TAU = DEFAULT_TIME_DECAY_SECONDS
DEPTH = DEFAULT_DEPTH
BLEND = DEFAULT_BLEND


# --- brute-force reference: the ORIGINAL O(E^2) formulas, inline ------------

def bf_wl_kernel(g1, g2):
    h1, h2 = wl_relabel(g1, DEPTH), wl_relabel(g2, DEPTH)
    score = 0.0
    for it1, it2 in zip(h1, h2):
        c1, c2 = Counter(it1.values()), Counter(it2.values())
        for label, n in c1.items():
            score += n * c2.get(label, 0)
    return float(score)


def bf_edge_kernel(g1, g2):
    by_label = {}
    for e in g1.edges.values():
        by_label.setdefault(_edge_label(e, g1), []).append(e)
    score = 0.0
    for e2 in g2.edges.values():
        for e1 in by_label.get(_edge_label(e2, g2), ()):
            dt = abs((e1.timestamp - e2.timestamp).total_seconds())
            score += math.exp(-dt / TAU)
    return float(score)


def bf_norm(k12, k11, k22):
    denom = math.sqrt(k11 * k22)
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, k12 / denom))


def bf_structural(live, pattern_rebased):
    wl = bf_norm(bf_wl_kernel(live, pattern_rebased),
                 bf_wl_kernel(live, live),
                 bf_wl_kernel(pattern_rebased, pattern_rebased))
    ed = bf_norm(bf_edge_kernel(live, pattern_rebased),
                 bf_edge_kernel(live, live),
                 bf_edge_kernel(pattern_rebased, pattern_rebased))
    combined = BLEND * wl + (1.0 - BLEND) * ed
    return max(0.0, min(1.0, combined)), wl, ed


# --- comparison -------------------------------------------------------------

FIXTURES = [
    ("redundant_cycle", lambda: fx_redundant_cycle(rounds=4)),
    ("injection_plus_cycle", fx_both),
    ("many_agents_400", lambda: fx_many_agents(20, 20)),
    ("wide_siblings_50", lambda: fx_wide_siblings(50)),
    ("deep_120", lambda: fx_deep(120)),
    ("big_flat_300", lambda: fx_big_flat(300)),
]

TOL = 1e-9


def main():
    library = builtin_library()
    worst = 0.0
    checks = 0
    decision_mismatches = 0

    for name, gen in FIXTURES:
        spans = gen()
        live = build_dynamic_graph(spans)
        matcher = AttackPathMatcher(library=library)

        live_anchor = _earliest_timestamp(live)
        from openweave_core.sentinel_agent.wl_kernel import (
            wl_label_counters, wl_kernel_from_counters,
            edge_label_buckets, edge_self_kernel_from_buckets,
        )
        live_wl_counters = wl_label_counters(live, depth=DEPTH)
        live_wl_self = wl_kernel_from_counters(live_wl_counters, live_wl_counters)
        live_buckets = edge_label_buckets(live)
        live_edge_self = edge_self_kernel_from_buckets(live_buckets, time_decay_seconds=TAU)

        for path in library:
            opt = matcher._structural_score(
                path,
                live_anchor=live_anchor,
                live_wl_counters=live_wl_counters,
                live_wl_self=live_wl_self,
                live_buckets=live_buckets,
                live_edge_self=live_edge_self,
            )
            rebased = _rebase_pattern_timestamps(path.pattern_graph, live_anchor)
            ref_combined, ref_wl, ref_ed = bf_structural(live, rebased)

            for label, a, b in (
                ("combined", opt.combined, ref_combined),
                ("wl", opt.wl, ref_wl),
                ("edge", opt.edge_distance, ref_ed),
            ):
                d = abs(a - b)
                worst = max(worst, d)
                checks += 1
                if d > TOL:
                    print(f"  MISMATCH {name}/{path.id}/{label}: opt={a:.12f} ref={b:.12f} d={d:.2e}")
                # threshold decision must agree
            if (opt.combined >= matcher._struct_thr) != (ref_combined >= matcher._struct_thr):
                decision_mismatches += 1
                print(f"  DECISION MISMATCH {name}/{path.id}: opt={opt.combined:.6f} ref={ref_combined:.6f}")

        print(f"  [ok] {name:24s} edges={len(live.edges):5d} "
              f"paths={len(list(library)):2d} checked")

    print("-" * 60)
    print(f"checks: {checks}   worst |d|: {worst:.2e}   "
          f"threshold-decision mismatches: {decision_mismatches}")
    if worst <= TOL and decision_mismatches == 0:
        print("PASS — optimized kernel is numerically identical to brute force.")
        return 0
    print("FAIL — divergence detected.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
