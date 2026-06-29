"""Time run_sentinel_agent across span counts — isolates the kernel cliff.

Sentinel's default judges are pure regex (no network), so this wall time is
all CPU. Before the kernel fix, big_flat ~3000 spans took ~22 min; after, it
should be well under a second.

Run from openweave-observabily/:  python scripts/probe_sentinel_scale.py
"""
import sys
import time

sys.path.insert(0, ".")

from openweave_core.anomaly_pipeline.contracts import TraceBundle
from openweave_core.anomaly_pipeline.runners import run_sentinel_agent

from scripts.stress_test import fx_big_flat, fx_many_agents, fx_wide_siblings


def time_case(name, spans):
    bundle = TraceBundle.from_spans(f"t-{name}", spans)
    t = time.perf_counter()
    result = run_sentinel_agent(bundle)
    dt = (time.perf_counter() - t) * 1000.0
    n_find = result.raw_summary.get("n_findings", "?")
    sev = result.raw_summary.get("overall_severity", "?")
    print(f"  {name:22s} spans={len(spans):5d}  sentinel={dt:9.1f} ms  "
          f"findings={n_find}  severity={sev}")


def main():
    print("Sentinel timing (pure CPU — heuristic judges, no network):")
    for n in (300, 800, 1500, 3000):
        time_case(f"big_flat_{n}", fx_big_flat(n))
    time_case("wide_siblings_500", fx_wide_siblings(500))
    time_case("many_agents_2000", fx_many_agents(100, 20))


if __name__ == "__main__":
    main()
