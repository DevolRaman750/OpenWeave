"""profile_cycle.py — attribute the cycle_detection cliff to a function."""
from __future__ import annotations
import cProfile, pstats, io, sys
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

spans = flat(200)
# Profile the *structural* stages only (no network embedding), which is where
# the 3000-span run burned 165 CPU-seconds.
def work():
    ct = sort_call_stack(spans)
    cands = detect_cycles(ct)
    flagged = [s for c in cands for s in c.first_occurrence]
    build_dag_and_siblings(spans, flagged)

pr = cProfile.Profile(); pr.enable(); work(); pr.disable()
s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(12)
print(s.getvalue())
