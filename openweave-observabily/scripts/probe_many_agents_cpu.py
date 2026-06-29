"""Isolate the cycle_detection CPU cost from embedding network latency.

Runs the full structural path (sort -> CDCS -> DAG) on the many_agents_2000
fixture, then confirm_cycles() with a STUB embed client that returns instant
deterministic vectors. Wall time here = pure algorithm cost, no network.
If this is sub-second, the 404s seen in the live stress run was network wait
on the single batched embedding call (today's degraded NVIDIA endpoint), not
an algorithmic cliff.
"""
import sys, time, hashlib
sys.path.insert(0, ".")

from scripts.stress_test import fx_many_agents
from openweave_core.cycle_detection import (
    sort_call_stack,
    detect_cycles,
    build_dag_and_siblings,
    confirm_cycles,
)


class StubEmbed:
    """Deterministic 32-d vector per text, no network. Counts calls/texts."""
    def __init__(self):
        self.calls = 0
        self.total_texts = 0

    def embed(self, texts):
        self.calls += 1
        self.total_texts += len(texts)
        out = []
        for t in texts:
            if not t.strip():
                out.append([])
                continue
            h = hashlib.md5(t.encode()).digest()
            out.append([b / 255.0 for b in h[:32]])
        return out


def main():
    spans = fx_many_agents(100, 20)
    print(f"spans: {len(spans)}")

    t = time.perf_counter()
    ct = sort_call_stack(spans)
    t_sort = time.perf_counter() - t

    t = time.perf_counter()
    candidates = detect_cycles(ct)
    t_cdcs = time.perf_counter() - t

    seen = set()
    flagged = []
    for c in candidates:
        for s in c.first_occurrence:
            sid = getattr(s, "id", None)
            if sid in seen:
                continue
            seen.add(sid)
            flagged.append(s)

    t = time.perf_counter()
    _, groups = build_dag_and_siblings(spans, flagged)
    t_dag = time.perf_counter() - t

    stub = StubEmbed()
    t = time.perf_counter()
    result = confirm_cycles(groups, embed_client=stub)
    t_confirm = time.perf_counter() - t

    print(f"candidates:      {len(candidates)}")
    print(f"sibling_groups:  {len(groups)}")
    print(f"embed calls:     {stub.calls}  (texts embedded: {stub.total_texts})")
    print(f"label:           {result.label}  confirmed_pairs: {len(result.confirmed_pairs)}")
    print("-" * 50)
    print(f"sort_call_stack:        {t_sort*1000:8.1f} ms")
    print(f"detect_cycles (CDCS):   {t_cdcs*1000:8.1f} ms")
    print(f"build_dag_and_siblings: {t_dag*1000:8.1f} ms")
    print(f"confirm_cycles (stub):  {t_confirm*1000:8.1f} ms")
    print(f"TOTAL cycle_detection:  {(t_sort+t_cdcs+t_dag+t_confirm)*1000:8.1f} ms")


if __name__ == "__main__":
    main()
