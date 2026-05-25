"""
aggregation.py — AMDM Step 3: Per-Axis Aggregation, EWMA Baselines,
                 and Axis-Level Anomaly Detection.

Consumes the per-span ``NormalizedMetricVector`` from Step 2 and emits an
``AxisEvaluation`` that:

  * aggregates per-axis z-scores into a single axis score S_A(t),
  * maintains a per-axis adaptive baseline θ_A(t) and rolling stdev σ_S_A(t)
    using the EWMA formulas from IDEA.md §B and the paper's amdm.py,
  * flags an axis anomaly when |S_A(t) - θ_A(t)| > k · σ_S_A(t),
  * reports exactly which axes triggered the alert for this span.

Formulas
--------
For every axis A at time t:

    S_A(t)      = mean( z_i(t)  for non-None metrics i in A )           ← aggregate
    θ_A(t)      = λ · S_A(t)   + (1 - λ) · θ_A(t-1)                     ← EWMA mean
    σ²_S_A(t)   = λ · (S_A(t) - θ_A(t))² + (1 - λ) · σ²_S_A(t-1)         ← EWMA variance
    flag        = |S_A(t) - θ_A(t)| > k · σ_S_A(t)                       ← detection

Notes
-----
* λ = 0.25 (paper recommendation, balances reactivity and stability).
* k = 3.0 (configurable per the user spec).
* σ uses the EWMA-of-squared-deviations approximation (cheap, O(1), matches
  ``Adaptive-Multi-Dimensional-Monitoring/amdm.py``) — not a separate rolling
  window. This keeps Step 3 fully streaming.
* Missing-data handling: if every metric on an axis is None for a given
  span, S_A(t) is reported as None, the EWMA state is **not** updated, and
  no anomaly is raised for that axis.
* Warm-up: the first observation initialises the baseline to the score and
  σ to a numerical floor (1e-6). To avoid spurious flags driven by the
  early near-zero σ, anomalies are suppressed for the first
  ``WARMUP_STEPS`` axis observations (configurable).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from openweave_core.adaptive_baseline.normalization import (
    AXIS_FIELDS,
    NormalizedMetricVector,
)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

LAMBDA_DEFAULT: float = 0.25  # paper recommendation (IDEA.md §B)
K_DEFAULT: float = 3.0        # sensitivity multiplier (user spec)
WARMUP_STEPS_DEFAULT: int = 10
_EPS: float = 1e-6            # numerical floor for σ to avoid divide-by-zero

# Canonical ordering of axes — used everywhere a deterministic order matters
# (joint vector for Step 4, dashboards, serialisation).
AXIS_ORDER: tuple[str, ...] = (
    "capability",
    "robustness",
    "safety",
    "human",
    "economic",
)

# Mapping axis-name → attribute name on NormalizedMetricVector.
# (Identity today; kept explicit so the two layers can evolve independently.)
_AXIS_ATTR: dict[str, str] = {a: a for a in AXIS_ORDER}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class AxisScore:
    """Per-axis evaluation for one span.

    Attributes
    ----------
    name:                 axis identifier (e.g. ``"capability"``).
    score:                S_A(t) — mean of observed z-scores on this axis.
                          ``None`` when no metric on this axis was observed.
    baseline:             θ_A(t) — current EWMA baseline.
                          ``None`` if this axis has never been observed.
    std:                  σ_S_A(t) — current EWMA-derived stdev.
    deviation:            |S_A(t) - θ_A(t)|, ``None`` if no observation.
    threshold:            k · σ_S_A(t), ``None`` if no observation.
    anomaly:              True iff deviation > threshold AND past warm-up.
    contributing_metrics: how many non-None z-scores fed the aggregation.
    """

    name: str
    score: Optional[float] = None
    baseline: Optional[float] = None
    std: Optional[float] = None
    deviation: Optional[float] = None
    threshold: Optional[float] = None
    anomaly: bool = False
    contributing_metrics: int = 0


@dataclass
class AxisEvaluation:
    """Step-3 output for one span — feeds the dashboard and Step 4.

    Attributes
    ----------
    span_id, trace_id, timestamp : carried through from upstream.
    axes:           one AxisScore per axis, keyed by axis name.
    any_anomaly:    True iff at least one axis raised an anomaly flag.
    triggered_axes: ordered list of axis names that flagged.
    """

    span_id: str
    trace_id: str
    timestamp: Optional[datetime] = None
    axes: Dict[str, AxisScore] = field(default_factory=dict)
    any_anomaly: bool = False
    triggered_axes: list[str] = field(default_factory=list)

    # Convenience accessor
    def axis_score(self, name: str) -> Optional[float]:
        """Return S_A for axis *name*, or None if no observation."""
        axis = self.axes.get(name)
        return axis.score if axis else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "any_anomaly": self.any_anomaly,
            "triggered_axes": list(self.triggered_axes),
            "axes": {n: ax.__dict__ for n, ax in self.axes.items()},
        }


# ---------------------------------------------------------------------------
# Per-axis adaptive baseline (EWMA mean + EWMA variance)
# ---------------------------------------------------------------------------

class EWMABaseline:
    """Streaming adaptive baseline for one axis score.

    State per instance: ``θ`` (EWMA mean), ``var`` (EWMA variance of
    deviations from θ), and ``n`` (number of observations so far).

    Memory: O(1) per axis. No window — variance is an exponentially-weighted
    estimate, matching ``Adaptive-Multi-Dimensional-Monitoring/amdm.py``.
    """

    __slots__ = ("_lambda", "_theta", "_variance", "_n")

    def __init__(self, lambda_: float = LAMBDA_DEFAULT) -> None:
        if not 0.0 < lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in (0, 1], got {lambda_}")
        self._lambda = lambda_
        self._theta: Optional[float] = None
        self._variance: float = 0.0
        self._n: int = 0

    # -- API ---------------------------------------------------------------

    def update(self, score: float) -> Tuple[float, float]:
        """Ingest a new axis score; return ``(new_theta, new_std)``.

        On the first call the baseline is seeded with *score* and the std is
        set to the numerical floor ``_EPS``. After that the EWMA recurrences
        from IDEA.md §B run normally.
        """
        self._n += 1

        if self._theta is None:
            self._theta = float(score)
            self._variance = 0.0
            return self._theta, _EPS

        lam = self._lambda
        # IMPORTANT ordering (matches the paper): θ is updated *first*, then
        # the variance is computed against the *new* θ. The reference impl
        # `amdm.py:152-156` does this — keep parity for numerical fidelity.
        theta = lam * float(score) + (1.0 - lam) * self._theta
        self._variance = lam * (float(score) - theta) ** 2 + (1.0 - lam) * self._variance
        self._theta = theta

        std = max(self._variance ** 0.5, _EPS)
        return theta, std

    # -- diagnostics -------------------------------------------------------

    @property
    def observations(self) -> int:
        return self._n

    @property
    def theta(self) -> Optional[float]:
        return self._theta

    @property
    def std(self) -> float:
        return max(self._variance ** 0.5, _EPS) if self._n > 0 else _EPS


# ---------------------------------------------------------------------------
# Main aggregator — public API
# ---------------------------------------------------------------------------

class AxisAggregator:
    """Step-3 driver: NormalizedMetricVector → AxisEvaluation.

    Owns one ``EWMABaseline`` per axis. Stateful, not thread-safe.

    Parameters
    ----------
    lambda_       : EWMA smoothing parameter, default 0.25 (paper).
    k             : anomaly sensitivity multiplier, default 3.0.
    warmup_steps  : number of axis observations to silently accumulate before
                    anomalies are flagged. Prevents early-life σ undershoot
                    from producing spurious alerts.

    Examples
    --------
    >>> agg = AxisAggregator(k=3.0)
    >>> for zvec in stream:                       # zvec: NormalizedMetricVector
    ...     ev = agg.evaluate(zvec)
    ...     if ev.any_anomaly:
    ...         alert(ev.span_id, ev.triggered_axes)
    """

    def __init__(
        self,
        *,
        lambda_: float = LAMBDA_DEFAULT,
        k: float = K_DEFAULT,
        warmup_steps: int = WARMUP_STEPS_DEFAULT,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self._k = k
        self._warmup = warmup_steps
        self._baselines: dict[str, EWMABaseline] = {
            axis: EWMABaseline(lambda_) for axis in AXIS_ORDER
        }

    # -- public ------------------------------------------------------------

    def evaluate(self, zvec: NormalizedMetricVector) -> AxisEvaluation:
        """Aggregate, update EWMA, and run axis-anomaly detection."""
        if not isinstance(zvec, NormalizedMetricVector):
            raise TypeError(
                f"zvec must be NormalizedMetricVector, got {type(zvec).__name__}"
            )

        per_axis: dict[str, AxisScore] = {}
        triggered: list[str] = []

        for axis_name in AXIS_ORDER:
            score, n_contrib = self._aggregate_axis(zvec, axis_name)
            per_axis[axis_name] = self._update_and_flag(axis_name, score, n_contrib)
            if per_axis[axis_name].anomaly:
                triggered.append(axis_name)

        return AxisEvaluation(
            span_id=zvec.span_id,
            trace_id=zvec.trace_id,
            timestamp=zvec.timestamp,
            axes=per_axis,
            any_anomaly=bool(triggered),
            triggered_axes=triggered,
        )

    def evaluate_many(
        self, vectors: list[NormalizedMetricVector]
    ) -> list[AxisEvaluation]:
        """Evaluate a batch in arrival order (each call updates state)."""
        return [self.evaluate(v) for v in vectors]

    # -- internal ----------------------------------------------------------

    def _aggregate_axis(
        self, zvec: NormalizedMetricVector, axis_name: str
    ) -> tuple[Optional[float], int]:
        """Return ``(S_A(t), n_contributing_metrics)`` for one axis.

        S_A is the arithmetic mean of the non-None z-scores on this axis.
        Returns ``(None, 0)`` if every metric on the axis is missing.
        """
        sub = getattr(zvec, _AXIS_ATTR[axis_name])
        observed: list[float] = []
        for field_name in AXIS_FIELDS[axis_name]:
            value = getattr(sub, field_name, None)
            if value is not None:
                observed.append(float(value))

        if not observed:
            return None, 0
        if len(observed) == 1:
            return observed[0], 1
        return statistics.mean(observed), len(observed)

    def _update_and_flag(
        self, axis_name: str, score: Optional[float], n_contrib: int
    ) -> AxisScore:
        """Apply EWMA update + threshold check for one axis."""
        baseline = self._baselines[axis_name]

        # No observation this step → leave EWMA untouched, report neutral.
        if score is None:
            return AxisScore(
                name=axis_name,
                score=None,
                baseline=baseline.theta,
                std=baseline.std if baseline.observations > 0 else None,
                deviation=None,
                threshold=None,
                anomaly=False,
                contributing_metrics=0,
            )

        theta, std = baseline.update(score)
        deviation = abs(score - theta)
        threshold = self._k * std

        # Suppress flags during warm-up — σ is still settling.
        flag = (
            baseline.observations > self._warmup
            and deviation > threshold
        )

        return AxisScore(
            name=axis_name,
            score=score,
            baseline=theta,
            std=std,
            deviation=deviation,
            threshold=threshold,
            anomaly=flag,
            contributing_metrics=n_contrib,
        )

    # -- diagnostics -------------------------------------------------------

    @property
    def k(self) -> float:
        return self._k

    @property
    def warmup_steps(self) -> int:
        return self._warmup

    def baseline_states(self) -> dict[str, tuple[Optional[float], float, int]]:
        """``{axis: (θ, σ, n_observations)}`` — for dashboards/debugging."""
        return {
            axis: (b.theta, b.std, b.observations)
            for axis, b in self._baselines.items()
        }
