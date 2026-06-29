"""OpenWeave continuous observer.

Watches a Langfuse project for traces emitted by real agents / multi-agent
systems and runs the OpenWeave anomaly + classification (+ deep-eval) pipeline
on each one automatically, persisting results to disk and/or the UI.

Run as a service:  ``python -m openweave_core.observer``
"""

from __future__ import annotations

from openweave_core.observer.config import ObserverConfig
from openweave_core.observer.daemon import ObserverStats, TraceObserver
from openweave_core.observer.sink import (
    CompositeSink,
    FileSink,
    LogSink,
    Sink,
    TrpcSink,
    build_sink,
)
from openweave_core.observer.state import ObserverState

__all__ = [
    "ObserverConfig",
    "ObserverState",
    "ObserverStats",
    "TraceObserver",
    "Sink",
    "LogSink",
    "FileSink",
    "TrpcSink",
    "CompositeSink",
    "build_sink",
]
