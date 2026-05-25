"""
normalization.py — AMDM Step 2: Rolling Z-Score Normalisation.

Consumes the raw heterogeneous MetricVector from Step 1 and emits a
NormalizedMetricVector whose every field is a unit-free z-score, grouped
under the same five axes.

Formula
-------
For every metric m_i at time t:

    z_i(t) = ( m_i(t) - mu_i(t) ) / sigma_i(t)

where mu_i(t) and sigma_i(t) are the rolling mean and standard deviation
over the last w observed values (w = WINDOW_SIZE = 80, per IDEA.md).

Conventions
-----------
*   None inputs (e.g. cost not measured) are *skipped* — they are not added
    to the rolling window and the output z-score for that field stays None.
*   When the window has fewer than 2 observed values, or stdev is exactly
    zero (all-identical window), the z-score is returned as 0.0 — the
    "no anomaly" baseline. This keeps the downstream Step-3 axis aggregation
    free of None-handling logic.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Optional

from openweave_core.adaptive_baseline.metrics import MetricVector

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

WINDOW_SIZE: int = 80  # IDEA.md §A — covers typical agent-task cycle length

# Mapping from MetricVector field names to their axis group. Defines the
# *complete* set of metrics the normaliser tracks; adding a new metric in
# Step 1 means adding it here too.
AXIS_FIELDS: dict[str, tuple[str, ...]] = {
    "capability": ("latency", "total_tokens", "success"),
    "economic":   ("cost",),
    "robustness": ("tool_diversity", "tool_drift_score", "new_tool_introduced"),
    "safety":     ("safety_score",),
    "human":      ("user_feedback",),
}


# ---------------------------------------------------------------------------
# Per-axis dataclasses — self-documenting structured output
# ---------------------------------------------------------------------------

@dataclass
class CapabilityZ:
    """Axis 1 — Capability & Efficiency z-scores."""
    latency: Optional[float] = None
    total_tokens: Optional[float] = None
    success: Optional[float] = None


@dataclass
class EconomicZ:
    """Axis 2 — Economic & Sustainability z-scores."""
    cost: Optional[float] = None


@dataclass
class RobustnessZ:
    """Axis 3 — Robustness & Adaptability z-scores."""
    tool_diversity: Optional[float] = None
    tool_drift_score: Optional[float] = None
    new_tool_introduced: Optional[float] = None


@dataclass
class SafetyZ:
    """Axis 4 — Safety & Ethics z-scores."""
    safety_score: Optional[float] = None


@dataclass
class HumanZ:
    """Axis 5 — Human-Centred Interaction z-scores."""
    user_feedback: Optional[float] = None


@dataclass
class NormalizedMetricVector:
    """Step-2 output: 5-axis z-score vector for a single span.

    The Step-3 axis aggregator consumes one of these per span; the EWMA and
    Mahalanobis layers depend on these being unit-free scalars per axis.
    """

    # Identity (carried through from MetricVector)
    span_id: str
    trace_id: str
    timestamp: Optional[datetime] = None

    # 5 axes
    capability: CapabilityZ = field(default_factory=CapabilityZ)
    economic: EconomicZ = field(default_factory=EconomicZ)
    robustness: RobustnessZ = field(default_factory=RobustnessZ)
    safety: SafetyZ = field(default_factory=SafetyZ)
    human: HumanZ = field(default_factory=HumanZ)

    def as_dict(self) -> dict[str, Any]:
        """Nested dict view — convenient for serialisation / debug printing."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "capability": self.capability.__dict__,
            "economic": self.economic.__dict__,
            "robustness": self.robustness.__dict__,
            "safety": self.safety.__dict__,
            "human": self.human.__dict__,
        }

    def flat(self) -> dict[str, Optional[float]]:
        """Flat {field: z} view across all axes — handy for assertions/tests."""
        out: dict[str, Optional[float]] = {}
        for fields in AXIS_FIELDS.values():
            for f in fields:
                # Locate which axis sub-dataclass owns `f` and read it.
                for axis in (
                    self.capability, self.economic, self.robustness,
                    self.safety, self.human,
                ):
                    if hasattr(axis, f):
                        out[f] = getattr(axis, f)
                        break
        return out


# ---------------------------------------------------------------------------
# Per-metric rolling z-scorer
# ---------------------------------------------------------------------------

class RollingZScorer:
    """Fixed-window rolling z-score for a single metric.

    The current value is added to the window *before* computing mu/sigma —
    this is the convention used in the AMDM paper and matches the formula
    in IDEA.md §A where m_i(t) is part of the rolling stats at time t.

    Parameters
    ----------
    window_size:
        Maximum number of observed (non-None) values retained.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self._window: Deque[float] = deque(maxlen=window_size)

    def update(self, value: Optional[float]) -> Optional[float]:
        """Add *value* to the window and return its rolling z-score.

        Returns
        -------
        Optional[float]
            * ``None`` if *value* is ``None`` (no observation to score).
            * ``0.0`` if the window has fewer than 2 entries or zero variance
              (cold start / constant signal — no anomaly by definition).
            * z-score otherwise.
        """
        if value is None:
            return None

        v = float(value)
        self._window.append(v)
        n = len(self._window)

        if n < 2:
            return 0.0

        mu = statistics.mean(self._window)
        try:
            sigma = statistics.stdev(self._window)
        except statistics.StatisticsError:
            return 0.0

        if sigma == 0.0:
            return 0.0

        return (v - mu) / sigma

    # -- diagnostics -------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._window)

    @property
    def stats(self) -> tuple[float, float]:
        """``(mean, stdev)`` of the current window; ``(0, 0)`` if n < 2."""
        if len(self._window) < 2:
            return (0.0, 0.0)
        try:
            return (statistics.mean(self._window), statistics.stdev(self._window))
        except statistics.StatisticsError:
            return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Main normaliser — public API
# ---------------------------------------------------------------------------

class Normalizer:
    """Stateful 5-axis rolling z-score normaliser.

    Owns one ``RollingZScorer`` per metric tracked in ``AXIS_FIELDS``. Every
    call to ``normalize`` updates each per-metric window with that span's
    value (or skips it if the value is ``None``) and emits a fully populated
    ``NormalizedMetricVector``.

    Thread-safety
    -------------
    Not thread-safe; wrap externally if you need concurrent ingest.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._scorers: dict[str, RollingZScorer] = {
            field_name: RollingZScorer(window_size)
            for fields in AXIS_FIELDS.values()
            for field_name in fields
        }

    # -- public ------------------------------------------------------------

    def normalize(self, vector: MetricVector) -> NormalizedMetricVector:
        """Normalise one ``MetricVector`` into z-scores for every metric."""
        if not isinstance(vector, MetricVector):
            raise TypeError(
                f"vector must be MetricVector, got {type(vector).__name__}"
            )

        z = {
            field_name: self._scorers[field_name].update(getattr(vector, field_name))
            for fields in AXIS_FIELDS.values()
            for field_name in fields
        }

        return NormalizedMetricVector(
            span_id=vector.span_id,
            trace_id=vector.trace_id,
            timestamp=vector.timestamp,
            capability=CapabilityZ(
                latency=z["latency"],
                total_tokens=z["total_tokens"],
                success=z["success"],
            ),
            economic=EconomicZ(cost=z["cost"]),
            robustness=RobustnessZ(
                tool_diversity=z["tool_diversity"],
                tool_drift_score=z["tool_drift_score"],
                new_tool_introduced=z["new_tool_introduced"],
            ),
            safety=SafetyZ(safety_score=z["safety_score"]),
            human=HumanZ(user_feedback=z["user_feedback"]),
        )

    def normalize_many(self, vectors: list[MetricVector]) -> list[NormalizedMetricVector]:
        """Normalise a batch in arrival order (each call updates state)."""
        return [self.normalize(v) for v in vectors]

    # -- diagnostics -------------------------------------------------------

    def window_sizes(self) -> dict[str, int]:
        """``{metric_name: current_window_size}`` — useful during warm-up."""
        return {name: scorer.size for name, scorer in self._scorers.items()}

    def metric_stats(self) -> dict[str, tuple[float, float]]:
        """``{metric_name: (mean, stdev)}`` — current rolling stats."""
        return {name: scorer.stats for name, scorer in self._scorers.items()}
