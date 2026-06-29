"""redundant_scale.py — the headline case: a genuinely redundant loop at high
span counts must be detected in well under a second (exact-match, no embedding)."""
from __future__ import annotations
import sys, time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openweave_core.anomaly_pipeline import run_anomaly_pipeline_for_spans_full
from openweave_core.models.span import ParsedSpan

T0 = datetime(2026, 1, 1)
def redundant(rounds):
    """`rounds` x (plan -> retrieve -> synth) with IDENTICAL outputs == a loop."""
    root = ParsedSpan(id="s0", trace_id="t", span_type="SPAN", tool_name="research-agent",
                      timestamp=T0, start_time=T0, end_time=T0)
    spans=[root]; i=1
    for r in range(rounds):
        for tn, out in (("plan-step","search the same corpus again"),
                        ("retrieve","doc-14: compliance memo (same as before)"),
                        ("synthesize","Q3 compliance held steady.")):
            ts=T0+timedelta(seconds=i)
            spans.append(ParsedSpan(id=f"s{i}", trace_id="t",
                span_type="GENERATION" if tn!="retrieve" else "SPAN", tool_name=tn,
                parent_id="s0", input_text=f"{tn} in", output_text=out,
                latency=.1, cost=1e-4, timestamp=ts, start_time=ts, end_time=ts))
            i+=1
    return spans

print(f"{'rounds':>7}{'spans':>7}{'wall_s':>9}  cycle_detection / flagged / severity")
for rounds in (50, 200, 1000):  # 1000 rounds = 3001 spans
    spans = redundant(rounds)
    t=time.perf_counter()
    rep,_ = run_anomaly_pipeline_for_spans_full(spans, trace_id="t")
    dt=time.perf_counter()-t
    cyc=next(d for d in rep.detector_results if d.detector=="cycle_detection")
    print(f"{rounds:>7}{len(spans):>7}{dt:>9.3f}  cyc={cyc.duration_seconds:.3f}s "
          f"flags={cyc.flag_count} sev={rep.overall_severity.value} "
          f"confirmed={cyc.raw_summary.get('n_confirmed_pairs')}", flush=True)
