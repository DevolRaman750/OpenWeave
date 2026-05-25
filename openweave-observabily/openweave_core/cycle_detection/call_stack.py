"""
call_stack.py — Chronological ordering of parsed agent spans into a Call Stack (CT).

Formal invariant
----------------
For the output sequence  CT = ⟨s_π(1), s_π(2), …, s_π(n)⟩  the following holds:

    t(s_π(1)) ≤ t(s_π(2)) ≤ … ≤ t(s_π(n))

where  t(s)  is the creation timestamp of span  s.

Algorithm
---------
Primary  : ``sort_call_stack``       — Python's built-in ``sorted()`` (Timsort).
                                       O(n log n) worst-case; O(n) on nearly-sorted input.
Reference: ``merge_sort_call_stack`` — Pure recursive merge sort, O(n log n) / O(n).

Both functions share the same ``_span_key`` extractor, so their outputs are identical.

Timestamp resolution order (ParsedSpan *and* dict inputs)
----------------------------------------------------------
1. ``timestamp``  /  ``"timestamp"``
2. ``start_time`` /  ``"start_time"``
3. Neither present → span is appended *after* all timed spans (stable relative order).

Timezone constraint
-------------------
All timestamps in a single trace must be either all timezone-aware or all
timezone-naive.  Mixing the two raises ``TypeError`` during comparison.
Normalise to UTC before calling if your data source returns mixed offsets.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Union

from openweave_core.models.span import ParsedSpan

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SpanLike = Union[ParsedSpan, dict[str, Any]]

# Internal pairing: (original span, pre-computed sort key).
# Pre-computing keys reduces key-function evaluations from O(n log n) to O(n).
_SortKey = tuple[int, datetime]
_Keyed = tuple[SpanLike, _SortKey]

# Sentinel for spans with no usable timestamp.
# Priority bit 1 > 0 guarantees untimed spans sort *after* all timed spans
# without ever comparing datetime values against datetime.min.
_UNTIMED: _SortKey = (1, datetime.min)


# ---------------------------------------------------------------------------
# Key extractor
# ---------------------------------------------------------------------------

def _span_key(span: SpanLike) -> _SortKey:
    """Return a (priority, datetime) sort key for *span*.

    Priority 0 → timed span (participates in the formal time-ordering invariant).
    Priority 1 → untimed span (appended after all timed spans, stable order kept).

    String timestamps are parsed via ``datetime.fromisoformat``.
    """
    if isinstance(span, ParsedSpan):
        raw: Any = span.timestamp or span.start_time
    else:
        raw = span.get("timestamp") or span.get("start_time")

    if raw is None:
        return _UNTIMED

    if isinstance(raw, str):
        raw = datetime.fromisoformat(raw)

    return (0, raw)


# ---------------------------------------------------------------------------
# Pure merge sort — reference implementation
# ---------------------------------------------------------------------------

def _merge(left: list[_Keyed], right: list[_Keyed]) -> list[_Keyed]:
    """Merge two sorted keyed lists into one sorted keyed list.

    Uses pre-computed keys, so no key function is called here.

    Time : O(|left| + |right|)
    Space: O(|left| + |right|)  (result buffer)
    """
    result: list[_Keyed] = []
    i = j = 0
    while i < len(left) and j < len(right):
        # Stable: left element wins on tie (preserves original relative order).
        if left[i][1] <= right[j][1]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    # Append remaining elements (at most one branch has leftovers).
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def _merge_sort_impl(keyed: list[_Keyed]) -> list[_Keyed]:
    """Recursive divide-and-conquer merge sort on pre-keyed spans.

    Base case : n <= 1  → already sorted, return as-is.
    Inductive : split at midpoint, sort each half, merge.

    Time : O(n log n)  — log n levels, O(n) work per level
    Space: O(n)        — merge buffer; O(log n) call-stack frames
    """
    n = len(keyed)
    if n <= 1:
        return keyed
    mid = n >> 1  # floor(n / 2)
    left = _merge_sort_impl(keyed[:mid])
    right = _merge_sort_impl(keyed[mid:])
    return _merge(left, right)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sort_call_stack(
    spans: list[SpanLike],
    *,
    warn_on_missing_timestamps: bool = True,
) -> list[SpanLike]:
    """Sort *spans* chronologically to form the Call Stack sequence CT.

    Uses Python's built-in ``sorted()`` (Timsort) — the production-optimised
    path.  Timsort exploits natural runs in nearly-sorted trace data, achieving
    O(n) time on such inputs while guaranteeing O(n log n) in the worst case.

    Parameters
    ----------
    spans:
        Flat list of ``ParsedSpan`` objects or span dicts for a single trace.
        The list is **not** mutated; a new sorted list is returned.
    warn_on_missing_timestamps:
        Emit a ``RuntimeWarning`` when any span carries no usable timestamp.
        Those spans are appended after all timed spans in their original order.

    Returns
    -------
    list[SpanLike]
        New list satisfying t(s_pi(1)) <= t(s_pi(2)) <= ... <= t(s_pi(n)).

    Complexity
    ----------
    Time : O(n log n) worst-case; O(n) on nearly-sorted input (Timsort)
    Space: O(n) auxiliary  (key array + result list)
    """
    if warn_on_missing_timestamps:
        missing = sum(1 for s in spans if _span_key(s)[0] == 1)
        if missing:
            warnings.warn(
                f"{missing} span(s) carry no usable timestamp and will be "
                "appended at the end of CT in their original order.",
                RuntimeWarning,
                stacklevel=2,
            )

    # sorted() calls _span_key exactly once per element (key is cached internally),
    # so key-extraction cost is O(n), not O(n log n).
    return sorted(spans, key=_span_key)


def merge_sort_call_stack(
    spans: list[SpanLike],
    *,
    warn_on_missing_timestamps: bool = True,
) -> list[SpanLike]:
    """Sort *spans* chronologically using an explicit recursive merge sort.

    Produces an output identical to ``sort_call_stack``; use this variant when
    an explicit divide-and-conquer implementation is required (e.g. algorithm
    tracing, benchmarking, or environments without CPython's Timsort guarantee).

    Keys are pre-computed in O(n) before the sort begins, so the merge steps
    perform only cheap tuple comparisons — no repeated key-function calls.

    Parameters
    ----------
    spans:
        Flat list of ``ParsedSpan`` objects or span dicts for a single trace.
    warn_on_missing_timestamps:
        Same semantics as in ``sort_call_stack``.

    Returns
    -------
    list[SpanLike]
        New list satisfying t(s_pi(1)) <= t(s_pi(2)) <= ... <= t(s_pi(n)).

    Complexity
    ----------
    Time : O(n log n)  — n comparisons across each of log2 n merge levels
    Space: O(n)        — merge buffer; O(log n) call-stack depth
    """
    if warn_on_missing_timestamps:
        missing = sum(1 for s in spans if _span_key(s)[0] == 1)
        if missing:
            warnings.warn(
                f"{missing} span(s) carry no usable timestamp and will be "
                "appended at the end of CT in their original order.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Pre-compute all keys — O(n) — before the O(n log n) sort begins.
    keyed: list[_Keyed] = [(s, _span_key(s)) for s in spans]
    sorted_keyed = _merge_sort_impl(keyed)
    return [span for span, _ in sorted_keyed]
