"""
cdcs.py — Cycle Detection using Call Stack representation (CDCS).

Pipeline
--------
CT (sorted spans)
  -> Step 1 : sliding window  -> all contiguous subsequences, m > 2
  -> Step 2 : frequency map   -> w(S) per unique subsequence
  -> Step 3 : baseline        -> mu, sigma over frequency multiset F
  -> Step 4 : threshold       -> flag w(S) > mu + k*sigma
  -> list[CycleCandidate]     -> semantic similarity stage
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Union

from openweave_core.models.span import ParsedSpan

SpanLike = Union[ParsedSpan, dict[str, Any]]
SubseqKey = tuple[str, ...]  # ordered op-name signature, hashable

# Max repeated-pattern length we enumerate. Redundant agent loops are short
# motifs (plan->act->observe etc.); enumerating windows up to n makes
# _build_frequency_map O(n^3) (the scaling cliff). Capping to a small window
# makes it O(n * W) ~ linear, and any longer repeated block is still detected
# via its length<=W sub-windows, so recall is preserved. See scripts/scale_sweep.py.
MAX_WINDOW_SIZE: int = 10


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class CycleCandidate:
    """One flagged subsequence — passed downstream to semantic similarity."""

    signature: SubseqKey          # op names forming the pattern
    frequency: int                # w(S) — how many times S appears in CT
    length: int                   # window size m
    first_index: int              # start position in CT (0-based)
    first_occurrence: list[SpanLike] = field(default_factory=list)  # actual spans


# ---------------------------------------------------------------------------
# Step 1 helpers
# ---------------------------------------------------------------------------

def _op_name(span: SpanLike) -> str:
    """Canonical operation label for a span (tool_name preferred over span_type)."""
    if isinstance(span, ParsedSpan):
        return span.tool_name or span.span_type
    return (
        span.get("tool_name")
        or span.get("span_type")
        or span.get("op")
        or "unknown"
    )


def _build_frequency_map(
    ct: list[SpanLike],
    max_window: int = MAX_WINDOW_SIZE,
) -> tuple[Counter[SubseqKey], dict[SubseqKey, int]]:
    """Sliding window (3 <= m <= max_window) over CT -> (frequency, first-seen index).

    Window size is capped at ``max_window`` so the cost is O(n * max_window)
    (~linear) instead of O(n^3); keys pre-built as op-name tuples for O(1) hashing.
    """
    n = len(ct)
    ops: list[str] = [_op_name(s) for s in ct]  # materialise once — O(n)

    freq: Counter[SubseqKey] = Counter()
    first_seen: dict[SubseqKey, int] = {}

    upper = min(n, max_window)
    for m in range(3, upper + 1):      # window sizes: 3 ... min(n, max_window)
        for i in range(n - m + 1):     # start positions: 0 ... n-m
            key: SubseqKey = tuple(ops[i : i + m])
            freq[key] += 1
            if key not in first_seen:
                first_seen[key] = i

    return freq, first_seen


# ---------------------------------------------------------------------------
# Step 3 helper
# ---------------------------------------------------------------------------

def _compute_threshold(frequencies: list[int], k: float) -> float:
    """mu + k*sigma; falls back to mean when sigma is undefined (single unique subseq)."""
    if len(frequencies) < 2:
        return float(frequencies[0]) if frequencies else 0.0
    mu = statistics.mean(frequencies)
    sigma = statistics.stdev(frequencies)
    return mu + k * sigma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_cycles(ct: list[SpanLike]) -> list[CycleCandidate]:
    """Run CDCS on a chronologically sorted Call Stack.

    Parameters
    ----------
    ct:
        Output of ``sort_call_stack`` / ``merge_sort_call_stack`` — must be
        sorted ascending by timestamp before calling.

    Returns
    -------
    list[CycleCandidate]
        Flagged subsequences where w(S) > mu + k*sigma, ordered by frequency
        descending (highest-confidence cycles first).
        Empty list when CT has fewer than 3 spans or no subsequence exceeds
        the threshold.

    Raises
    ------
    TypeError
        If *ct* is not a list.
    """
    if not isinstance(ct, list):
        raise TypeError(f"ct must be a list, got {type(ct).__name__}")

    if len(ct) < 3:  # m > 2 requires at least 3 spans
        return []

    # Step 1 + 2
    freq_map, first_seen = _build_frequency_map(ct)

    if not freq_map:
        return []

    # Step 3
    frequencies = list(freq_map.values())
    k = 0.5  # tune here — 0.5 yields best F1-score for CDCS
    threshold = _compute_threshold(frequencies, k)

    # Step 4
    candidates: list[CycleCandidate] = []
    for key, count in freq_map.items():
        if count > threshold:
            idx = first_seen[key]
            candidates.append(
                CycleCandidate(
                    signature=key,
                    frequency=count,
                    length=len(key),
                    first_index=idx,
                    first_occurrence=ct[idx : idx + len(key)],
                )
            )

    candidates.sort(key=lambda c: c.frequency, reverse=True)
    return candidates
