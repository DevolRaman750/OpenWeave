"""
joint_anomaly.py — AMDM Step 4: Joint Anomaly Detection via Mahalanobis Distance.

Consumes per-span ``AxisEvaluation`` from Step 3, maintains an online estimate
of the 5-dimensional mean vector μ(t) and a *shrunk* covariance matrix Σ(t)
across the canonical axes, and flags spans whose joint Mahalanobis distance

    D²(t) = (S(t) - μ(t))ᵀ · Σ(t)⁻¹ · (S(t) - μ(t))

exceeds χ²₅(1-α). With α=0.01 this gives a ~1 % false-alarm rate
(threshold ≈ 15.0863).

Why shrinkage (and how we do it)
--------------------------------
The pure empirical covariance is rank-deficient until n exceeds the
dimension and is noisy / nearly-singular for small n — so Σ⁻¹ blows up
during the cold-start period (IDEA.md §4).

We use a Ledoit-Wolf-style blend toward a stable target:

    Σ_shrunk = δ · F + (1 - δ) · S_empirical
    F        = (trace(S_empirical) / d) · I              ← scaled identity
    δ(n,d)   = max(δ_floor, d / (n + d - 1))             ← adaptive intensity

Properties:
* δ = 1.0 on the first observation → Σ = identity → guaranteed invertible.
* δ → δ_floor (default 0.05) as n grows → we converge to the empirical
  estimate but keep a small permanent shrink that protects invertibility
  even under degenerate inputs.
* The target F preserves the average variance scale of the data, so D² is
  on the right magnitude from step 1.

Detection threshold
-------------------
χ²-distribution with d=5 degrees of freedom. Computed via scipy when
available; falls back to hardcoded χ²₅(0.99) ≈ 15.0863 for the default α.

Cold-start handling
-------------------
Flagging is suppressed for the first ``warmup_steps`` *complete* joint
observations (default 50). During warm-up the detector still maintains
state and returns D², but ``joint_anomaly`` remains False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from openweave_core.adaptive_baseline.aggregation import (
    AXIS_ORDER,
    AxisEvaluation,
)

# ---------------------------------------------------------------------------
# Tunables / constants
# ---------------------------------------------------------------------------

JOINT_DIM: int = 5                                 # |S(t)| — must match AXIS_ORDER
CHI2_5_0_99: float = 15.086272469388987            # χ²₅(0.99) — hardcoded fallback

DEFAULT_ALPHA: float = 0.01                        # ~1 % false-alarm rate
DEFAULT_SHRINKAGE_FLOOR: float = 0.05              # δ never falls below this
DEFAULT_WARMUP_STEPS: int = 50                     # IDEA.md §4 (50-100 trace floor)

_RIDGE: float = 1e-9                               # final numerical safety ridge
_EPS: float = 1e-12

assert len(AXIS_ORDER) == JOINT_DIM, (
    f"AXIS_ORDER length {len(AXIS_ORDER)} must match JOINT_DIM {JOINT_DIM}"
)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class JointAnomalyResult:
    """Step-4 output for one span.

    Attributes
    ----------
    span_id, trace_id, timestamp : carried through from Step 3.
    state_vector  : S(t) in canonical AXIS_ORDER.
    mean_vector   : μ(t) — online mean over complete observations so far.
    d2            : D²(t) — squared Mahalanobis distance.
    threshold     : χ²_d(1-α) cutoff.
    severity      : ratio D² / threshold; values > 1.0 indicate anomaly.
    joint_anomaly : True iff D² > threshold AND we are past warm-up.
    requires_review: dashboard-friendly alias of ``joint_anomaly``.
    n_observations: number of complete joint observations so far.
    shrinkage     : δ used to mix target ↔ empirical this step.
    in_warmup     : True while flagging is still suppressed.
    """

    span_id: str
    trace_id: str
    timestamp: Optional[datetime] = None

    state_vector: list[float] = field(default_factory=list)
    mean_vector: list[float] = field(default_factory=list)

    d2: float = 0.0
    threshold: float = CHI2_5_0_99
    severity: float = 0.0

    joint_anomaly: bool = False
    requires_review: bool = False

    n_observations: int = 0
    shrinkage: float = 1.0
    in_warmup: bool = True

    def as_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "state_vector": list(self.state_vector),
            "mean_vector": list(self.mean_vector),
            "d2": self.d2,
            "threshold": self.threshold,
            "severity": self.severity,
            "joint_anomaly": self.joint_anomaly,
            "requires_review": self.requires_review,
            "n_observations": self.n_observations,
            "shrinkage": self.shrinkage,
            "in_warmup": self.in_warmup,
        }


# ---------------------------------------------------------------------------
# Online (Welford) mean + covariance accumulator
# ---------------------------------------------------------------------------

class WelfordCovariance:
    """Numerically stable online mean and covariance via Welford's algorithm.

    Maintains:
      n   — number of observations
      μ   — running mean vector (shape (d,))
      M2  — Σ_i (x_i - μ_new)(x_i - μ_old)ᵀ  (shape (d, d))

    Sample covariance with Bessel's correction: cov = M2 / (n - 1).

    O(d²) per update, O(d²) memory.
    """

    __slots__ = ("_dim", "_n", "_mean", "_M2")

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._n = 0
        self._mean = np.zeros(dim, dtype=float)
        self._M2 = np.zeros((dim, dim), dtype=float)

    def update(self, x: np.ndarray) -> None:
        """Ingest one d-dim observation."""
        self._n += 1
        delta_old = x - self._mean
        self._mean += delta_old / self._n
        delta_new = x - self._mean
        # outer-product form: M2 += (x - μ_new) ⊗ (x - μ_old)
        self._M2 += np.outer(delta_new, delta_old)

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    def covariance(self) -> np.ndarray:
        """Sample covariance (Bessel-corrected). Returns zeros if n < 2."""
        if self._n < 2:
            return np.zeros((self._dim, self._dim), dtype=float)
        return self._M2 / (self._n - 1)


# ---------------------------------------------------------------------------
# χ² threshold helper
# ---------------------------------------------------------------------------

def _chi2_threshold(alpha: float, dim: int) -> float:
    """Return χ²_d(1-α). Uses scipy when available; otherwise the hardcoded
    χ²₅(0.99) constant for the documented default. Raises if neither applies.
    """
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.ppf(1.0 - alpha, df=dim))
    except ImportError:
        if dim == JOINT_DIM and abs(alpha - DEFAULT_ALPHA) < 1e-9:
            return CHI2_5_0_99
        raise RuntimeError(
            "scipy is not installed and no hardcoded χ² value exists for "
            f"dim={dim}, alpha={alpha}. Install scipy or pass "
            "chi2_threshold= explicitly to MahalanobisDetector."
        )


# ---------------------------------------------------------------------------
# Main detector — public API
# ---------------------------------------------------------------------------

class MahalanobisDetector:
    """Online joint-anomaly detector with adaptive shrinkage covariance.

    Parameters
    ----------
    alpha             : target false-alarm rate. Threshold = χ²_d(1-α).
    shrinkage_floor   : minimum δ retained even with large n.
    warmup_steps      : # complete observations before flagging is enabled.
    chi2_threshold    : optional explicit override of the χ² cut-off
                        (bypasses the scipy lookup entirely).

    Notes
    -----
    Not thread-safe; one detector instance per ingest worker.
    """

    def __init__(
        self,
        *,
        alpha: float = DEFAULT_ALPHA,
        shrinkage_floor: float = DEFAULT_SHRINKAGE_FLOOR,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        chi2_threshold: Optional[float] = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0.0 <= shrinkage_floor <= 1.0:
            raise ValueError(
                f"shrinkage_floor must be in [0, 1], got {shrinkage_floor}"
            )
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self._alpha = alpha
        self._shrinkage_floor = shrinkage_floor
        self._warmup = warmup_steps
        self._dim = JOINT_DIM
        self._welford = WelfordCovariance(self._dim)
        self._threshold: float = (
            chi2_threshold
            if chi2_threshold is not None
            else _chi2_threshold(alpha, self._dim)
        )
        self._identity = np.eye(self._dim)

    # -- public ------------------------------------------------------------

    def evaluate(self, ev: AxisEvaluation) -> JointAnomalyResult:
        """Incorporate one span's axis scores and return its joint result."""
        if not isinstance(ev, AxisEvaluation):
            raise TypeError(
                f"ev must be AxisEvaluation, got {type(ev).__name__}"
            )

        state_vec, complete = self._build_state_vector(ev)

        # Only fully-observed joint vectors update the online μ/Σ. Imputed
        # dimensions would bias the mean toward itself and shrink variance.
        if complete:
            self._welford.update(state_vec)

        n = self._welford.n

        # Brand-new detector, no μ to compare against yet.
        if n < 1:
            return JointAnomalyResult(
                span_id=ev.span_id,
                trace_id=ev.trace_id,
                timestamp=ev.timestamp,
                state_vector=state_vec.tolist(),
                mean_vector=[0.0] * self._dim,
                d2=0.0,
                threshold=self._threshold,
                severity=0.0,
                joint_anomaly=False,
                requires_review=False,
                n_observations=0,
                shrinkage=1.0,
                in_warmup=True,
            )

        mean = self._welford.mean
        S_emp = self._welford.covariance()
        delta = self._shrinkage_intensity(n)
        sigma = self._shrink(S_emp, delta)
        d2 = self._mahalanobis(state_vec, mean, sigma)

        in_warmup = n <= self._warmup
        flag = (not in_warmup) and (d2 > self._threshold)

        return JointAnomalyResult(
            span_id=ev.span_id,
            trace_id=ev.trace_id,
            timestamp=ev.timestamp,
            state_vector=state_vec.tolist(),
            mean_vector=mean.tolist(),
            d2=d2,
            threshold=self._threshold,
            severity=d2 / self._threshold,
            joint_anomaly=flag,
            requires_review=flag,
            n_observations=n,
            shrinkage=delta,
            in_warmup=in_warmup,
        )

    def evaluate_many(
        self, evaluations: list[AxisEvaluation]
    ) -> list[JointAnomalyResult]:
        """Evaluate a batch in arrival order (each call updates state)."""
        return [self.evaluate(e) for e in evaluations]

    # -- internal ----------------------------------------------------------

    def _build_state_vector(
        self, ev: AxisEvaluation
    ) -> tuple[np.ndarray, bool]:
        """Construct S(t) in AXIS_ORDER; impute None axes with running μ.

        Returns ``(vector, complete)``. ``complete`` is True iff every axis
        had a real (non-None) score. Only complete observations update
        Welford so the online mean stays unbiased.
        """
        running_mean = (
            self._welford.mean if self._welford.n > 0
            else np.zeros(self._dim)
        )
        vec = np.zeros(self._dim, dtype=float)
        complete = True
        for i, axis_name in enumerate(AXIS_ORDER):
            axis_score = ev.axes.get(axis_name)
            value = (
                axis_score.score
                if axis_score is not None and axis_score.score is not None
                else None
            )
            if value is None:
                # Substitute current μ_i so this dim contributes 0 to D².
                vec[i] = running_mean[i]
                complete = False
            else:
                vec[i] = float(value)
        return vec, complete

    def _shrinkage_intensity(self, n: int) -> float:
        """δ(n, d) = max(floor, d / (n + d - 1)). Hits 1.0 when n=1."""
        if n < 2:
            return 1.0
        adaptive = self._dim / (n + self._dim - 1)
        return float(max(self._shrinkage_floor, min(1.0, adaptive)))

    def _shrink(self, S_emp: np.ndarray, delta: float) -> np.ndarray:
        """Σ = δ · F + (1-δ) · S_emp,   F = (trace(S_emp)/d) · I.

        Degenerate inputs (all-zero S_emp, or δ ≥ 1) fall back to the identity
        target so the matrix is still invertible.
        """
        if delta >= 1.0:
            return self._identity.copy()
        trace_avg = float(np.trace(S_emp)) / self._dim
        if trace_avg < _EPS:
            return self._identity.copy()
        target = trace_avg * self._identity
        return delta * target + (1.0 - delta) * S_emp

    def _mahalanobis(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        sigma: np.ndarray,
    ) -> float:
        """Compute D² = (x-μ)ᵀ Σ⁻¹ (x-μ) robustly."""
        diff = x - mean
        # Tiny ridge for numerical safety on top of the shrinkage.
        sigma_safe = sigma + _RIDGE * self._identity
        try:
            # solve is more numerically stable than inv() + matmul.
            inv_diff = np.linalg.solve(sigma_safe, diff)
        except np.linalg.LinAlgError:
            inv_diff = np.linalg.pinv(sigma_safe) @ diff
        d2 = float(diff @ inv_diff)
        return max(0.0, d2)

    # -- diagnostics -------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def n_observations(self) -> int:
        return self._welford.n

    def state(self) -> dict:
        """Snapshot of internal state — for dashboards / debugging."""
        return {
            "n": self._welford.n,
            "mean": (
                self._welford.mean.tolist() if self._welford.n > 0 else None
            ),
            "cov_empirical": (
                self._welford.covariance().tolist()
                if self._welford.n > 1 else None
            ),
            "threshold": self._threshold,
            "alpha": self._alpha,
            "shrinkage_floor": self._shrinkage_floor,
            "warmup_steps": self._warmup,
            "in_warmup": self._welford.n <= self._warmup,
        }
