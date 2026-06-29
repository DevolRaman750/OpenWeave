"""struct_probe.py — time the structural cycle stages separately (no embedding)."""
from __future__ import annotations
import sys, time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openweave_core.cycle_detection import sort_call_stack, detect_cycles, build_dag_and_siblings
from openweave_core.models.span import ParsedSpan

T0 = datetime(2026, 1, 1)
def flat(n):
    root = ParsedSpan(id="s0", trace_id="t", span_type="SPAN", tool_name="root",
                      timestamp=T0, start_time=T0, end_time=T0)
    spans=[root]
    for i in range(1, n+1):
        ts=T0+timedelta(seconds=i)
        spans.append(ParsedSpan(id=f"s{i}", trace_id="t", span_type="SPAN",
            tool_name=f"tool_{i%200}", parent_id="s0", input_text=f"t{i}",
            output_text=f"o{i} distinct {i}", latency=.1, cost=1e-4,
            timestamp=ts, start_time=ts, end_time=ts))
    return spans

for n in (250, 500, 1000, 1500, 3000):
    spans = flat(n)
    t=time.perf_counter(); ct=sort_call_stack(spans); t_sort=time.perf_counter()-t
    t=time.perf_counter(); cands=detect_cycles(ct); t_cdcs=time.perf_counter()-t
    flagged=[s for c in cands for s in c.first_occurrence]
    t=time.perf_counter(); _,groups=build_dag_and_siblings(spans, flagged); t_dag=time.perf_counter()-t
    widths=sorted((len(g.siblings) for g in groups), reverse=True)[:3]
    print(f"n={n:>5}  sort={t_sort:6.3f}  cdcs={t_cdcs:7.3f}  dag={t_dag:6.3f}  "
          f"n_cands={len(cands):>4} n_flagged={len(flagged):>5} n_groups={len(groups):>3} top_widths={widths}", flush=True)
