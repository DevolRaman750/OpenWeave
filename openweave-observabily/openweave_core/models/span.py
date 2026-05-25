from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ParsedSpan:
    """Clean span shape produced from a raw Langfuse trace observation.

    The field names are distilled from the broader Langfuse `Span` model in the
    repository root while keeping the parser output small and ready for the
    observability pipeline.
    """

    id: str
    trace_id: str
    span_type: str
    tool_name: Optional[str] = None

    input_text: str = ""
    output_text: str = ""
    input_hash: str = ""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    latency: Optional[float] = None
    cost: Optional[float] = None
    timestamp: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    parent_id: Optional[str] = None
    child_ids: list[str] = field(default_factory=list)
    depth: int = 0

    model: Optional[str] = None
    metadata: Optional[Any] = None
    status_message: Optional[str] = None
    level: Optional[str] = None
