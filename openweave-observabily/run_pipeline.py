"""
run_pipeline.py — Single-command entry point for the Hybrid Cycle Detection pipeline.

Usage
-----
    cd openweave-observabily

    # auto-discover most recent Langfuse trace
    python run_pipeline.py

    # pin a specific trace
    python run_pipeline.py <trace_id>

Environment (resolved automatically from agents/.env)
------------------------------------------------------
    LANGFUSE_BASE_URL   http://127.0.0.1:3000   (use 127.0.0.1, not localhost)
    LANGFUSE_PUBLIC_KEY pk-lf-...
    LANGFUSE_SECRET_KEY sk-lf-...
    NVIDIA_API_KEY      nvapi-...               (loaded from cycle_detection/.env)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from base64 import b64encode

from openweave_core.parser import parse_langfuse_trace
from openweave_core.cycle_detection import (
    sort_call_stack,
    detect_cycles,
    build_dag_and_siblings,
    confirm_cycles,
)


def _op(span) -> str:
    return span.tool_name or span.span_type or "?"


def _discover_trace_id() -> str:
    base_url = (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or "https://cloud.langfuse.com"
    ).rstrip("/")
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pub or not sec:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")

    auth = b64encode(f"{pub}:{sec}".encode()).decode()
    req = urllib.request.Request(
        f"{base_url}/api/public/traces?limit=5",
        headers={"Authorization": f"Basic {auth}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    traces = data.get("data") or []
    if not traces:
        raise RuntimeError("No traces found in Langfuse")
    return traces[0]["id"]


def main() -> int:
    print("=" * 70)
    print("  Hybrid Cycle Detection Pipeline")
    print("=" * 70)

    # ── Resolve trace_id ────────────────────────────────────────────────
    if len(sys.argv) > 1:
        trace_id = sys.argv[1]
        print(f"\n[trace]  {trace_id}  (provided)")
    else:
        try:
            trace_id = _discover_trace_id()
            print(f"\n[trace]  {trace_id}  (latest from Langfuse)")
        except Exception as exc:
            print(f"\n[FATAL] {exc}")
            return 2

    # ── Stage 0 : Parse ─────────────────────────────────────────────────
    print("\n[0] Parsing trace ...")
    try:
        spans = parse_langfuse_trace(trace_id)
    except Exception as exc:
        print(f"    [FATAL] {type(exc).__name__}: {exc}")
        return 2
    print(f"    {len(spans)} span(s) fetched")

    # ── Stage 1 : Call Stack ────────────────────────────────────────────
    print("\n[1] Building chronological Call Stack CT ...")
    ct = sort_call_stack(spans)
    ops_preview = " -> ".join(_op(s) for s in ct[:10])
    if len(ct) > 10:
        ops_preview += " ..."
    print(f"    |CT| = {len(ct)}:  {ops_preview}")

    # ── Stage 2 : CDCS ──────────────────────────────────────────────────
    print("\n[2] CDCS — frequency baseline (k=0.5) ...")
    candidates = detect_cycles(ct)
    print(f"    {len(candidates)} candidate(s) flagged")
    for i, c in enumerate(candidates[:5], 1):
        print(f"      #{i}  w(S)={c.frequency}  m={c.length}  {c.signature}")
    if not candidates:
        print("\n" + "=" * 70)
        print("  VERDICT:  f(T) = 0   ->   CLEAN  (no cycle detected)")
        print("=" * 70)
        return 0

    # ── Stage 3 : DAG + siblings ────────────────────────────────────────
    print("\n[3] Building DAG and sibling groups ...")
    flagged = [s for c in candidates for s in c.first_occurrence]
    dag, groups = build_dag_and_siblings(spans, flagged)
    print(f"    nodes={len(dag.nodes)}  edges={len(dag.edge_weights)}  "
          f"roots={len(dag.roots)}  groups={len(groups)}")

    # ── Stage 4 : Semantic confirmation ─────────────────────────────────
    print("\n[4] Semantic confirmation — nv-embedcode-7b-v1  (phi=0.83) ...")
    try:
        result = confirm_cycles(groups)
    except Exception as exc:
        print(f"    [FATAL] {type(exc).__name__}: {exc}")
        return 2

    # ── Verdict ─────────────────────────────────────────────────────────
    verdict = "BAD CYCLE — redundant loop confirmed" if result.label == 1 else "CLEAN"
    print("\n" + "=" * 70)
    print(f"  VERDICT:  f(T) = {result.label}   ->   {verdict}")
    print("=" * 70)

    if result.confirmed_pairs:
        print(f"\nTop confirmed pairs  ({len(result.confirmed_pairs)} total):")
        for i, p in enumerate(result.confirmed_pairs[:8], 1):
            a, b = _op(p.span_a), _op(p.span_b)
            print(f"  #{i:>2}  cos={p.similarity:.4f}   {a} <-> {b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
