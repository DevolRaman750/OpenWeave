"""
metrics.py — AMDM Step 1: Metric Extraction & Mapping.

Converts a single ParsedSpan into a 5-axis raw metric vector ready to feed
into Step 2 (z-score normalization). Stateful, because Axis 3 (Robustness)
needs a per-trace + global baseline of tool-name variations.

Axes
----
    1. Capability & Efficiency   (S_cap)  latency, total_tokens, success
    2. Economic & Sustainability (S_eco)  cost
    3. Robustness & Adaptability (S_rob)  tool diversity + drift score
    4. Safety & Ethics           (S_saf)  external judge over output_text
    5. Human-Centred Interaction (S_hum)  user feedback in metadata

Cold-start note
---------------
* Axis 3 drift score returns 0.0 until at least ``BASELINE_MIN_SAMPLES``
  traces have been finalised (rolling baseline can't be computed before).
* Axis 4 returns ``None`` whenever no SafetyEvaluator has been wired in;
  the Step-2 normalizer is expected to skip None-valued metrics.

Usage
-----
    extractor = MetricExtractor(safety_evaluator=my_judge)

    for span in parse_langfuse_trace(trace_id):
        vector = extractor.extract(span)
        normalizer.feed(vector)        # Step 2 consumer

    extractor.finalize_trace(trace_id) # promotes diversity to global baseline
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Optional, Protocol, runtime_checkable

from openweave_core.models.span import ParsedSpan

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

BASELINE_HISTORY_SIZE: int = 100   # how many completed traces inform the baseline
BASELINE_MIN_SAMPLES: int = 5      # min samples before drift score becomes non-zero

_ERROR_LEVELS: frozenset[str] = frozenset({"ERROR", "FATAL", "CRITICAL"})
_ERROR_KEYWORDS: tuple[str, ...] = (
    "error", "failed", "failure", "exception",
    "timeout", "timed out", "fatal", "crash", "aborted",
)

# Keys we recognise as user-feedback signals inside ``ParsedSpan.metadata``.
# First match wins.
_FEEDBACK_KEYS: tuple[str, ...] = (
    "user_rating", "user_feedback", "rating",
    "trust_score", "feedback_score", "satisfaction",
    "thumbs", "score",
)

_THUMBS_UP_VALUES: frozenset[str] = frozenset(
    {"up", "thumbs_up", "positive", "good", "yes", "y", "true", "1"}
)
_THUMBS_DOWN_VALUES: frozenset[str] = frozenset(
    {"down", "thumbs_down", "negative", "bad", "no", "n", "false", "0"}
)


# ---------------------------------------------------------------------------
# Safety evaluator interface
# ---------------------------------------------------------------------------

@runtime_checkable
class SafetyEvaluator(Protocol):
    """Plug-point for a secondary judge (LLM, classifier, rule engine).

    Implementations should return a score in ``[0.0, 1.0]`` where 1.0 = safest /
    most grounded and 0.0 = severe hallucination / toxicity. Return ``None`` if
    the evaluator cannot score this particular span (e.g. empty output).
    """

    def score(self, output_text: str, *, span: ParsedSpan) -> Optional[float]:
        ...


class NullSafetyEvaluator:
    """Default no-op evaluator — emits ``None`` until a real judge is wired in."""

    def score(self, output_text: str, *, span: ParsedSpan) -> Optional[float]:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Output dataclass — the Step-1 → Step-2 contract
# ---------------------------------------------------------------------------

@dataclass
class MetricVector:
    """Raw heterogeneous metric vector for a single span.

    Mixed units are intentional: Step 2 (z-score normalization) is the layer
    that homogenises them. Optional fields signal "no measurement available"
    rather than zero, so the normalizer can skip rather than skew.
    """

    # Identity
    span_id: str
    trace_id: str
    timestamp: Optional[datetime] = None

    # Axis 1 — Capability & Efficiency
    latency: Optional[float] = None      # seconds
    total_tokens: int = 0                # compute usage proxy
    success: float = 1.0                 # binary: 1.0 ok | 0.0 failed

    # Axis 2 — Economic & Sustainability
    cost: Optional[float] = None         # currency units (USD typical)

    # Axis 3 — Robustness & Adaptability (goal drift)
    tool_diversity: int = 0              # |distinct tools in trace so far|
    tool_drift_score: float = 0.0        # spikes when diversity ≫ historical baseline
    new_tool_introduced: float = 0.0     # 1.0 if this span uses a tool unseen in trace

    # Axis 4 — Safety & Ethics
    safety_score: Optional[float] = None # 0.0 unsafe … 1.0 safe; None if no judge

    # Axis 5 — Human-Centred Interaction
    user_feedback: Optional[float] = None  # e.g. 0.0 thumbs-down … 1.0 thumbs-up

    def as_dict(self) -> dict[str, Any]:
        """Flat dict view — convenient for the Step-2 rolling-window ingest."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            # Axis 1
            "latency": self.latency,
            "total_tokens": self.total_tokens,
            "success": self.success,
            # Axis 2
            "cost": self.cost,
            # Axis 3
            "tool_diversity": self.tool_diversity,
            "tool_drift_score": self.tool_drift_score,
            "new_tool_introduced": self.new_tool_introduced,
            # Axis 4
            "safety_score": self.safety_score,
            # Axis 5
            "user_feedback": self.user_feedback,
        }


# ---------------------------------------------------------------------------
# Axis 3 — Goal-drift baseline
# ---------------------------------------------------------------------------

class ToolVariationBaseline:
    """Tracks tool-name usage per trace + a rolling global diversity baseline.

    State
    -----
    _active   : trace_id  →  Counter[tool_name]  (open traces; tools seen so far)
    _baseline : rolling list of final unique-tool counts from finalised traces

    The drift score for an *active* trace is the one-sided z-score of its
    current unique-tool count against the global baseline distribution.
    The score is clamped at zero (we only care about positive drift — the
    agent reaching for *more* tools than usual).
    """

    def __init__(self, history_size: int = BASELINE_HISTORY_SIZE) -> None:
        self._active: dict[str, Counter[str]] = {}
        self._baseline: Deque[int] = deque(maxlen=history_size)

    def record(
        self,
        trace_id: str,
        tool_name: Optional[str],
    ) -> tuple[int, bool]:
        """Register a tool call. Returns (unique_count_in_trace, was_new_to_trace).

        ``tool_name`` may be ``None`` (non-tool spans like generations); in
        that case the diversity counter is *not* incremented but the trace is
        still registered so finalisation works.
        """
        bucket = self._active.setdefault(trace_id, Counter())
        if not tool_name:
            return len(bucket), False

        was_new = tool_name not in bucket
        bucket[tool_name] += 1
        return len(bucket), was_new

    def drift_score(self, trace_id: str) -> float:
        """Positive z-score of current trace diversity vs. global baseline.

        Returns 0.0 during cold start (< BASELINE_MIN_SAMPLES finalised traces)
        or when the baseline has zero variance.
        """
        current = len(self._active.get(trace_id, ()))
        if len(self._baseline) < BASELINE_MIN_SAMPLES:
            return 0.0

        mu = statistics.mean(self._baseline)
        try:
            sigma = statistics.stdev(self._baseline)
        except statistics.StatisticsError:
            return 0.0
        if sigma <= 0.0:
            return 0.0

        return max(0.0, (current - mu) / sigma)

    def finalize(self, trace_id: str) -> None:
        """Mark a trace complete; promote its final diversity to the baseline."""
        bucket = self._active.pop(trace_id, None)
        if bucket is not None:
            self._baseline.append(len(bucket))

    def baseline_stats(self) -> tuple[int, float, float]:
        """Return ``(n_samples, mean, stdev)`` for diagnostics / dashboards."""
        n = len(self._baseline)
        if n < 2:
            return (n, 0.0, 0.0)
        return (n, statistics.mean(self._baseline), statistics.stdev(self._baseline))


# ---------------------------------------------------------------------------
# Per-axis extractors (free functions — pure, easy to unit-test)
# ---------------------------------------------------------------------------

def _extract_success(span: ParsedSpan) -> float:
    """Binary success: 0.0 on detected error signal, 1.0 otherwise.

    Decision order:
        1. ``level`` in {ERROR, FATAL, CRITICAL}  → 0.0
        2. ``status_message`` contains an error keyword (case-insensitive) → 0.0
        3. otherwise → 1.0
    """
    if span.level and span.level.upper() in _ERROR_LEVELS:
        return 0.0
    if span.status_message:
        msg = span.status_message.lower()
        if any(keyword in msg for keyword in _ERROR_KEYWORDS):
            return 0.0
    return 1.0


def _extract_user_feedback(metadata: Any) -> Optional[float]:
    """Extract a numeric feedback value from a metadata dict.

    Recognises numeric ratings, booleans, and stringified thumbs up/down.
    Returns ``None`` if nothing recognisable is present.
    """
    if not isinstance(metadata, dict):
        return None

    for key in _FEEDBACK_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        coerced = _coerce_feedback_value(value)
        if coerced is not None:
            return coerced
    return None


def _coerce_feedback_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return None
        if token in _THUMBS_UP_VALUES:
            return 1.0
        if token in _THUMBS_DOWN_VALUES:
            return 0.0
        try:
            return float(token)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Main extractor — public API
# ---------------------------------------------------------------------------

class MetricExtractor:
    """Stateful per-span metric extractor for AMDM Step 1.

    Owns the goal-drift baseline (Axis 3) and the safety-evaluator plug-in
    (Axis 4). Every other axis is computed purely from the span and is
    stateless.

    Thread-safety
    -------------
    Not thread-safe. Wrap calls externally (e.g. one extractor per worker,
    merge baselines periodically) if you need concurrency.
    """

    def __init__(
        self,
        *,
        safety_evaluator: Optional[SafetyEvaluator] = None,
        baseline_history_size: int = BASELINE_HISTORY_SIZE,
    ) -> None:
        self._safety: SafetyEvaluator = safety_evaluator or NullSafetyEvaluator()
        self._tool_baseline = ToolVariationBaseline(baseline_history_size)

    # -- public ----------------------------------------------------------

    def extract(self, span: ParsedSpan) -> MetricVector:
        """Produce the raw 5-axis vector for a single span."""
        if not isinstance(span, ParsedSpan):
            raise TypeError(f"span must be ParsedSpan, got {type(span).__name__}")

        # Axis 3 state update + signals (do this first; later axes are pure)
        diversity, was_new = self._tool_baseline.record(span.trace_id, span.tool_name)
        drift = self._tool_baseline.drift_score(span.trace_id)

        # Axis 4 — defer to plug-in
        safety: Optional[float] = None
        if span.output_text:
            try:
                safety = self._safety.score(span.output_text, span=span)
            except Exception:
                # A misbehaving judge must not break extraction.
                safety = None

        return MetricVector(
            span_id=span.id,
            trace_id=span.trace_id,
            timestamp=span.timestamp,
            # Axis 1
            latency=span.latency,
            total_tokens=int(span.total_tokens or 0),
            success=_extract_success(span),
            # Axis 2
            cost=span.cost,
            # Axis 3
            tool_diversity=diversity,
            tool_drift_score=drift,
            new_tool_introduced=1.0 if was_new else 0.0,
            # Axis 4
            safety_score=safety,
            # Axis 5
            user_feedback=_extract_user_feedback(span.metadata),
        )

    def extract_many(self, spans: list[ParsedSpan]) -> list[MetricVector]:
        """Convenience: extract a batch in arrival order."""
        return [self.extract(s) for s in spans]

    def finalize_trace(self, trace_id: str) -> None:
        """Promote a completed trace's diversity to the global baseline.

        Call this once after you've extracted every span for ``trace_id``.
        Forgetting to call it leaks memory in the active-trace map but does
        not corrupt downstream metrics.
        """
        self._tool_baseline.finalize(trace_id)

    # -- diagnostics -----------------------------------------------------

    @property
    def baseline_stats(self) -> tuple[int, float, float]:
        """``(samples, mean, stdev)`` of the goal-drift baseline."""
        return self._tool_baseline.baseline_stats()
