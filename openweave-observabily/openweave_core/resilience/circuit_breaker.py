"""circuit_breaker.py — a small, dependency-free circuit breaker.

Why this exists
---------------
The pipeline calls external services (the NVIDIA embeddings endpoint, the
Anthropic judge). Per-call ``timeout`` + ``max_retries`` bound a *single* call,
but when a dependency is fully down they don't stop us from paying that timeout
on *every* trace. A circuit breaker adds that missing layer: after enough
consecutive failures it "trips" and rejects calls instantly for a cooldown
window, so a dead upstream fails in microseconds instead of seconds-per-trace.

State machine
-------------
    CLOSED      — normal. Calls pass through; failures are counted.
    OPEN        — tripped. Calls are rejected immediately with CircuitOpenError
                  until ``recovery_timeout`` elapses.
    HALF_OPEN   — probation. One trial call is allowed; success closes the
                  breaker, failure re-opens it for another cooldown.

``CircuitOpenError`` subclasses ``RuntimeError`` on purpose: the cycle-detection
degrade path already treats ``RuntimeError`` from the embedder as "API
unavailable -> fall back to structural/exact-match", so a tripped breaker
flows through the existing graceful-degradation logic with no extra wiring.

Thread-safety
-------------
Detectors run in worker threads (``asyncio.to_thread``) and the breaker is a
process-wide singleton guarding a shared endpoint, so all state transitions are
guarded by a ``threading.Lock``. The wrapped function runs *outside* the lock
(we never hold the lock across a network call).
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the breaker is OPEN.

    Subclasses RuntimeError so existing ``except RuntimeError`` degrade paths
    (e.g. ``confirm_cycles`` / ``run_cycle_detection``) treat a tripped breaker
    exactly like an upstream API failure.
    """


class CircuitBreaker:
    """Trips OPEN after ``failure_threshold`` consecutive failures.

    Parameters
    ----------
    name              : label used in error messages / introspection.
    failure_threshold : consecutive failures that trip the breaker (>= 1).
    recovery_timeout  : seconds to stay OPEN before allowing a HALF_OPEN trial.
    time_fn           : monotonic clock injection (tests pass a fake clock).
    """

    def __init__(
        self,
        *,
        name: str = "circuit",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = max(1, int(failure_threshold))
        self._recovery = float(recovery_timeout)
        self._time = time_fn

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0

    # -- introspection ---------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current state, re-evaluating the OPEN -> HALF_OPEN cooldown."""
        with self._lock:
            self._maybe_half_open()
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def reset(self) -> None:
        """Force the breaker back to CLOSED (operational override / tests)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = 0.0

    # -- core ------------------------------------------------------------

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn(*args, **kwargs)`` through the breaker.

        Raises ``CircuitOpenError`` immediately if the breaker is OPEN and still
        within its cooldown. Otherwise runs the call, recording success/failure.
        Any exception from ``fn`` counts as a failure and is re-raised unchanged.
        """
        with self._lock:
            if not self._allow_locked():
                raise CircuitOpenError(
                    f"circuit '{self.name}' is OPEN; failing fast "
                    f"(retry after ~{self._recovery:.0f}s cooldown)"
                )

        try:
            result = fn(*args, **kwargs)
        except BaseException:
            self._record_failure()
            raise
        self._record_success()
        return result

    # -- internals (must hold lock where noted) --------------------------

    def _maybe_half_open(self) -> None:
        """If OPEN and the cooldown has elapsed, move to HALF_OPEN. Hold lock."""
        if (
            self._state is CircuitState.OPEN
            and self._time() - self._opened_at >= self._recovery
        ):
            self._state = CircuitState.HALF_OPEN

    def _allow_locked(self) -> bool:
        """Whether a call may proceed right now. Hold lock."""
        self._maybe_half_open()
        # CLOSED and HALF_OPEN both let a call through; OPEN rejects.
        return self._state is not CircuitState.OPEN

    def _record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            # A failed HALF_OPEN trial, or hitting the threshold, trips OPEN.
            if (
                self._state is CircuitState.HALF_OPEN
                or self._consecutive_failures >= self._threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = self._time()
