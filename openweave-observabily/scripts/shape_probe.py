"""shape_probe.py — is the cycle_detection cliff driven by total spans or by
sibling-group WIDTH? Compare a wide (flat) trace vs a tree-shaped trace."""
from __future__ import annotations
import sys, time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openweave_core.anomaly_pipeline import run_anomaly_pipeline_for_spans_full
from openweave_core.models.span import ParsedSpan

T0 = datetime(2026, 1, 1)
_n = [0]
def sp(parent, i):
    ts = T0 + timedelta(seconds=i)
    return ParsedSpan(id=f"s{i}", trace_id="t", span_type="SPAN", tool_name=f"tool_{i%50}",
                      parent_id=parent, input_text=f"task {i} unique",
                      output_text=f"out {i} distinct {i*3}", latency=0.1, cost=1e-4,
                      timestamp=ts, start_time=ts, end_time=ts+timedelta(milliseconds=80))

def flat(n):
    root = sp(None, 0); root.tool_name = "root"
    return [root] + [sp("s0", i) for i in range(1, n+1)]

def tree(n, branch=4):
    # balanced tree: each node gets up to `branch` children -> max sibling width = branch
    root = sp(None, 0); root.tool_name = "root"
    spans = [root]; parents = ["s0"]; i = 1
    while i <= n:
        nextp = []
        for p in parents:
            for _ in range(branch):
                if i > n: break
                spans.append(sp(p, i)); nextp.append(f"s{i}"); i += 1
        parents = nextp or ["s0"]
    return spans

def timeit(spans, label):
    t0 = time.perf_counter()
    rep, _ = run_anomaly_pipeline_for_spans_full(spans, trace_id="t")
    dt = time.perf_counter() - t0
    cyc = next(d.duration_seconds for d in rep.detector_results if d.detector == "cycle_detection")
    print(f"  {label:<24} spans={len(spans):>5}  wall={dt:6.2f}s  cycle_detection={cyc:6.2f}s", flush=True)

print("Tree-shaped (max sibling width = 4):")
for n in (250, 1000, 3000):
    timeit(tree(n), f"tree n={n}")
print("Flat (one root, all siblings):")
for n in (150, 250):
    timeit(flat(n), f"flat n={n}")
