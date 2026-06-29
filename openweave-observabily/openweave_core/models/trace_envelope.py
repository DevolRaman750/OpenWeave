"""
trace_envelope.py — Trace-level metadata envelope.

The parser already fetches the full Langfuse trace document but used to drop
everything except the per-observation spans. Downstream layers — especially
incident classification and RAG routing — need trace-level context that has
no per-span equivalent:

    * the user's input prompt to the whole trace
    * the trace's final output
    * session/user identity
    * release / environment / tags
    * arbitrary trace-level metadata

``TraceEnvelope`` captures these fields. It is attached to ``TraceBundle`` so
all downstream consumers share one canonical view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TraceEnvelope:
    """Trace-level context — everything not on individual spans.

    Attributes
    ----------
    trace_id    : the trace identifier.
    name        : the trace's display name (often the root agent / task name).
    input_text  : serialized representation of the trace input (user prompt).
    output_text : serialized representation of the trace output (final answer).
    session_id  : conversation / session identifier the trace belongs to.
    user_id     : end-user identifier when known.
    release     : application release tag.
    version     : application version tag.
    environment : deployment environment (``"prod"``, ``"staging"``, …).
    tags        : list of free-form tags attached at trace creation.
    metadata    : free-form trace-level metadata (dict).
    timestamp   : when the trace began (UTC, when known).
    total_cost  : aggregate cost as reported by the backend, when present.
    raw         : the raw trace dict for downstream consumers that need fields
                  we haven't yet promoted to first-class attributes. Empty
                  when the envelope is constructed without a backing payload.
    """

    trace_id: str
    name: Optional[str] = None
    input_text: str = ""
    output_text: str = ""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    release: Optional[str] = None
    version: Optional[str] = None
    environment: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    total_cost: Optional[float] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, trace_id: str) -> "TraceEnvelope":
        """Construct a minimal envelope with only the trace_id set.

        Useful when the caller supplies spans directly (no fetch happened) but
        downstream code still needs the envelope shape.
        """
        return cls(trace_id=trace_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "release": self.release,
            "version": self.version,
            "environment": self.environment,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_cost": self.total_cost,
        }
