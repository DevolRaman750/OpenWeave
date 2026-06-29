"""Environment-driven configuration for the OpenWeave observer.

Everything the continuous observer needs is read from process environment
variables so it deploys cleanly as a 12-factor container. No flags are
required to start; the defaults are safe for a real deployment (eval is OFF by
default because the deep-eval judge costs money on every flagged trace).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> Optional[list[str]]:
    raw = os.environ.get(name)
    if not raw:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


@dataclass
class ObserverConfig:
    """Tunables for the continuous trace observer."""

    # How often to poll Langfuse for new agent traces.
    poll_interval_seconds: float = 15.0
    # Max traces to pull per page; the observer paginates until it catches up.
    page_limit: int = 50
    # Concurrent analyses in flight (bounds CPU + outbound API pressure).
    concurrency: int = 4
    # Run the live Claude deep-eval judge on flagged traces (costs money).
    run_eval: bool = False
    # On a cold start with no cursor, how far back to begin watching.
    lookback_seconds: int = 3600
    # Don't analyze a trace until it's this old, so all its spans have flushed.
    # Real multi-agent runs stream spans for seconds; analyzing too early
    # produces a partial graph.
    settle_seconds: int = 30
    # Per-tick wall-clock budget; stop pulling new pages past this.
    max_traces_per_tick: int = 500

    # Persistent cursor file (last processed timestamp + recent id ring).
    state_path: str = ".openweave_observer_state.json"

    # Optional server-side filters (scope to one environment / set of tags).
    tags: Optional[list[str]] = None
    environment: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Sinks.
    output_dir: Optional[str] = "openweave_reports"
    log_summaries: bool = True
    # Optional: POST each AnalysisPayload to the Langfuse/Node tRPC writer so
    # results show up in the OpenWeave UI. Requires an auth cookie/header.
    trpc_url: Optional[str] = None
    trpc_project_id: Optional[str] = None
    trpc_auth_header: Optional[str] = None

    # Labels for logs/metrics.
    service_name: str = "openweave-observer"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ObserverConfig":
        return cls(
            poll_interval_seconds=_env_float("OPENWEAVE_OBSERVE_INTERVAL", 15.0),
            page_limit=_env_int("OPENWEAVE_OBSERVE_LIMIT", 50),
            concurrency=_env_int("OPENWEAVE_OBSERVE_CONCURRENCY", 4),
            run_eval=_env_bool("OPENWEAVE_OBSERVE_RUN_EVAL", False),
            lookback_seconds=_env_int("OPENWEAVE_OBSERVE_LOOKBACK", 3600),
            settle_seconds=_env_int("OPENWEAVE_OBSERVE_SETTLE", 30),
            max_traces_per_tick=_env_int("OPENWEAVE_OBSERVE_MAX_PER_TICK", 500),
            state_path=os.environ.get(
                "OPENWEAVE_OBSERVE_STATE", ".openweave_observer_state.json"
            ),
            tags=_env_list("OPENWEAVE_OBSERVE_TAGS"),
            environment=os.environ.get("OPENWEAVE_OBSERVE_ENVIRONMENT") or None,
            user_id=os.environ.get("OPENWEAVE_OBSERVE_USER_ID") or None,
            session_id=os.environ.get("OPENWEAVE_OBSERVE_SESSION_ID") or None,
            output_dir=os.environ.get("OPENWEAVE_OBSERVE_OUTPUT_DIR", "openweave_reports")
            or None,
            log_summaries=_env_bool("OPENWEAVE_OBSERVE_LOG", True),
            trpc_url=os.environ.get("OPENWEAVE_OBSERVE_TRPC_URL") or None,
            trpc_project_id=os.environ.get("OPENWEAVE_OBSERVE_TRPC_PROJECT_ID") or None,
            trpc_auth_header=os.environ.get("OPENWEAVE_OBSERVE_TRPC_AUTH") or None,
        )
