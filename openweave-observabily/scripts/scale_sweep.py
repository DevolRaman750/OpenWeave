"""scale_sweep.py — find the span-count cliff and the culprit detector.

Runs a clean flat trace at increasing span counts, timing each detector
separately, and stops once a single trace exceeds the wall-clock budget.
"""
from __future__ import annotations
import sys, time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openweave_core.anomaly_pipeline import run_anomaly_pipeline_for_spans_full
from openweave_core.models.span import ParsedSpan

T0 = datetime(2026, 1, 1)
BUDGET_S = 30.0  # stop after a trace that takes longer than this


def flat(n: int):
    root = ParsedSpan(id="s0", trace_id="t", span_type="SPAN", tool_name="orchestrator",
                      timestamp=T0, start_time=T0, end_time=T0 + timedelta(milliseconds=50))
    spans = [root]
    for i in range(1, n + 1):
        ts = T0 + timedelta(seconds=i)
        spans.append(ParsedSpan(
            id=f"s{i}", trace_id="t", span_type="SPAN", tool_name=f"tool_{i % 200}",
            parent_id="s0", input_text=f"task {i} unique", output_text=f"out {i} distinct {i*3}",
            latency=0.1, cost=0.0001, timestamp=ts, start_time=ts,
            end_time=ts + timedelta(milliseconds=80)))
    return spans


print(f"{'n_spans':>8}{'wall_s':>9}   per-detector seconds")
for n in (100, 250, 500, 1000, 1500, 2000, 3000):
    spans = flat(n)
    t0 = time.perf_counter()
    report, _ = run_anomaly_pipeline_for_spans_full(spans, trace_id=f"t{n}")
    dt = time.perf_counter() - t0
    det = "  ".join(f"{d.detector}={d.duration_seconds:.2f}" for d in report.detector_results)
    print(f"{n:>8}{dt:>9.2f}   {det}", flush=True)
    if dt > BUDGET_S:
        print(f"\n>>> exceeded {BUDGET_S}s budget at n={n}; stopping sweep.")
        break
