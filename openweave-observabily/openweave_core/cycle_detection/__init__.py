"""
cycle_detection — Hybrid redundant-loop detection pipeline for agent traces.

Full pipeline (single call)
---------------------------
    from openweave_core.cycle_detection import run_detection
    result = run_detection("<trace_id>")
    print(result.label)          # 0 = clean | 1 = bad cycle
    print(result.confirmed_pairs)

Stages
------
  1. parse_langfuse_trace  — fetch raw spans from Langfuse
  2. sort_call_stack       — chronological CT (Timsort, O(n log n))
  3. detect_cycles         — CDCS sliding window + mu+k*sigma threshold
  4. build_dag_and_siblings — weighted DAG + sibling groups
  5. confirm_cycles        — cosine similarity via nv-embedcode-7b-v1 (phi=0.83)
"""

from openweave_core.cycle_detection.call_stack import sort_call_stack, merge_sort_call_stack
from openweave_core.cycle_detection.cdcs import detect_cycles, CycleCandidate
from openweave_core.cycle_detection.dag import (
    build_trace_dag,
    find_sibling_groups,
    build_dag_and_siblings,
    TraceDAG,
    SiblingGroup,
)
from openweave_core.cycle_detection.semantic import (
    confirm_cycles,
    SemanticResult,
    ConfirmedPair,
    EmbedClient,
)

__all__ = [
    # Call Stack
    "sort_call_stack",
    "merge_sort_call_stack",
    # CDCS
    "detect_cycles",
    "CycleCandidate",
    # DAG
    "build_trace_dag",
    "find_sibling_groups",
    "build_dag_and_siblings",
    "TraceDAG",
    "SiblingGroup",
    # Semantic
    "confirm_cycles",
    "SemanticResult",
    "ConfirmedPair",
    "EmbedClient",
    # Pipeline
    "run_detection",
]


def run_detection(trace_id: str) -> SemanticResult:
    """Run the full hybrid cycle-detection pipeline for one trace.

    Fetches the trace from Langfuse, sorts spans into CT, runs CDCS,
    builds the DAG, extracts sibling groups, and confirms cycles via
    semantic similarity.

    Parameters
    ----------
    trace_id:
        Langfuse trace ID to analyse.

    Returns
    -------
    SemanticResult
        label=0  → clean trace (no redundant loop detected)
        label=1  → bad cycle confirmed; see .confirmed_pairs for details
    """
    from openweave_core.parser import parse_langfuse_trace

    spans = parse_langfuse_trace(trace_id)
    ct = sort_call_stack(spans)
    candidates = detect_cycles(ct)

    if not candidates:
        return SemanticResult(label=0)

    flagged = [s for c in candidates for s in c.first_occurrence]
    _, groups = build_dag_and_siblings(spans, flagged)
    return confirm_cycles(groups)
