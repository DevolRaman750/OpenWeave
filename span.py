"""Langfuse Span (Observation) model — complete field coverage.

This module defines a comprehensive Span dataclass that mirrors every field
available in Langfuse's observation/span schema.  Fields are sourced from:

  • OpenWeave-SDK/langfuse/api/commons/types/observation.py        (v1 API)
  • OpenWeave-SDK/langfuse/api/commons/types/observation_v2.py     (v2 API)
  • OpenWeave-SDK/langfuse/api/commons/types/observations_view.py  (view model)
  • OpenWeave-SDK/langfuse/api/commons/types/usage.py              (legacy usage)
  • OpenWeave-SDK/langfuse/_client/span.py                         (SDK wrapper)
  • OpenWeave-SDK/langfuse/_client/attributes.py                   (OTEL attrs)
  • openweave-observabily/langfuse  Prisma schema                  (PostgreSQL)
  • openweave-observabily/langfuse  ClickHouse migrations          (analytics)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObservationType(str, enum.Enum):
    """All observation types supported by Langfuse.

    Source: OpenWeave-SDK/langfuse/_client/constants.py
    """

    # Span-like types (no model/usage fields)
    SPAN = "span"
    AGENT = "agent"
    TOOL = "tool"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    EVALUATOR = "evaluator"
    GUARDRAIL = "guardrail"

    # Generation-like types (have model/usage/cost fields)
    GENERATION = "generation"
    EMBEDDING = "embedding"

    # Point-in-time (no duration, cannot be updated after creation)
    EVENT = "event"


class ObservationLevel(str, enum.Enum):
    """Severity / importance level for an observation.

    Source: OpenWeave-SDK/langfuse/api/commons/types/observation_level.py
    """

    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Legacy Usage (deprecated — kept for backward compat)
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Deprecated legacy usage model.

    Source: OpenWeave-SDK/langfuse/api/commons/types/usage.py
    Langfuse recommends using ``usage_details`` and ``cost_details`` instead.
    """

    input: int = 0                       # Number of input units (e.g. tokens)
    output: int = 0                      # Number of output units
    total: int = 0                       # Defaults to input + output
    unit: Optional[str] = None           # Unit of measurement
    input_cost: Optional[float] = None   # USD input cost
    output_cost: Optional[float] = None  # USD output cost
    total_cost: Optional[float] = None   # USD total cost (defaults to input + output)


# ---------------------------------------------------------------------------
# Span (Observation) — complete field set
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """Complete Langfuse observation/span model.

    This dataclass represents the **union** of every field found across
    Langfuse's API response models, SDK wrapper classes, Prisma (PostgreSQL)
    schema, and ClickHouse analytics schema — so that no information is lost
    when parsing a span from a trace.

    Notes
    -----
    * Field names use ``snake_case`` matching the Python SDK convention.
    * ``input`` / ``output`` / ``metadata`` are ``Any`` because Langfuse
      stores them as arbitrary JSON (not just plain text).
    * Some fields are only populated in certain API versions or views.
    """

    # ------------------------------------------------------------------
    # Core identifiers
    # ------------------------------------------------------------------

    id: str = ""
    """Unique observation / span identifier.
    Source: Observation.id, ObservationV2.id, Prisma(id)"""

    trace_id: Optional[str] = None
    """ID of the parent trace this observation belongs to.
    Source: Observation.traceId, ObservationV2.traceId, Prisma(trace_id)"""

    parent_observation_id: Optional[str] = None
    """ID of the parent observation (for nested spans).
    Source: Observation.parentObservationId, ObservationV2.parentObservationId,
    Prisma(parent_observation_id)"""

    project_id: Optional[str] = None
    """Project this observation belongs to.
    Source: ObservationV2.projectId, Prisma(project_id), ClickHouse(project_id)"""

    # ------------------------------------------------------------------
    # Classification / type
    # ------------------------------------------------------------------

    type: str = ""
    """Observation type — one of ObservationType values.
    e.g. 'GENERATION', 'SPAN', 'EVENT', 'AGENT', 'TOOL', 'CHAIN',
    'RETRIEVER', 'EVALUATOR', 'EMBEDDING', 'GUARDRAIL'.
    Source: Observation.type, ObservationV2.type, Prisma(type)"""

    name: Optional[str] = None
    """Human-readable name for the observation (e.g. function name, LLM call).
    Source: Observation.name, ObservationV2.name, Prisma(name),
    ClickHouse(name)"""

    level: Optional[str] = None
    """Severity level: DEBUG | DEFAULT | WARNING | ERROR.
    Source: Observation.level, ObservationV2.level, Prisma(level)"""

    status_message: Optional[str] = None
    """Optional status / error message associated with the observation.
    Source: Observation.statusMessage, ObservationV2.statusMessage,
    Prisma(status_message)"""

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    start_time: Optional[datetime] = None
    """When the observation started.
    Source: Observation.startTime, ObservationV2.startTime,
    Prisma(start_time), ClickHouse(start_time)"""

    end_time: Optional[datetime] = None
    """When the observation ended.
    Source: Observation.endTime, ObservationV2.endTime,
    Prisma(end_time), ClickHouse(end_time)"""

    completion_start_time: Optional[datetime] = None
    """When the model started generating (for generation/embedding types).
    Used to derive time-to-first-token.
    Source: Observation.completionStartTime, ObservationV2.completionStartTime,
    Prisma(completion_start_time), ClickHouse(completion_start_time)"""

    latency: Optional[float] = None
    """Computed latency in **seconds** (end_time - start_time).
    Source: ObservationsView.latency, ObservationV2.latency"""

    time_to_first_token: Optional[float] = None
    """Time to first token in **seconds**
    (completion_start_time - start_time).
    Source: ObservationsView.timeToFirstToken,
    ObservationV2.timeToFirstToken"""

    # ------------------------------------------------------------------
    # I/O (arbitrary JSON, not just text)
    # ------------------------------------------------------------------

    input: Optional[Any] = None
    """Input data — can be any JSON-serialisable object.
    Source: Observation.input, ObservationV2.input,
    Prisma(input), ClickHouse(input)"""

    output: Optional[Any] = None
    """Output data — can be any JSON-serialisable object.
    Source: Observation.output, ObservationV2.output,
    Prisma(output), ClickHouse(output)"""

    metadata: Optional[Any] = None
    """Arbitrary metadata attached to the observation (JSON).
    Source: Observation.metadata, ObservationV2.metadata,
    Prisma(metadata), ClickHouse(metadata as Map)"""

    # ------------------------------------------------------------------
    # Model info (generation / embedding types)
    # ------------------------------------------------------------------

    model: Optional[str] = None
    """User-provided model name (e.g. "gpt-4", "claude-3-opus").
    Source: Observation.model, Prisma(model)"""

    provided_model_name: Optional[str] = None
    """Model name exactly as provided by the user (before Langfuse matching).
    Source: ObservationV2.providedModelName,
    ClickHouse(provided_model_name)"""

    internal_model: Optional[str] = None
    """Langfuse-matched internal model name (legacy, being deprecated).
    Source: Prisma(internal_model)"""

    internal_model_id: Optional[str] = None
    """Langfuse-matched internal model ID.
    Source: ObservationV2.internalModelId, Prisma(internal_model_id),
    ClickHouse(internal_model_id)"""

    model_id: Optional[str] = None
    """Matched model ID (used for pricing lookups).
    Source: ObservationsView.modelId, ObservationV2.modelId"""

    model_parameters: Optional[Any] = None
    """Parameters used for the model call (e.g. temperature, max_tokens).
    Stored as JSON.
    Source: Observation.modelParameters, ObservationV2.modelParameters,
    Prisma(model_parameters), ClickHouse(model_parameters)"""

    # ------------------------------------------------------------------
    # Usage details (new — recommended)
    # ------------------------------------------------------------------

    usage_details: Optional[Dict[str, int]] = None
    """Detailed token/unit usage as a map.
    e.g. {"input": 100, "output": 50, "total": 150}
    Source: Observation.usageDetails, ObservationV2.usageDetails,
    ClickHouse(usage_details)"""

    provided_usage_details: Optional[Dict[str, int]] = None
    """User-provided usage details (before Langfuse model matching).
    Source: ClickHouse(provided_usage_details)"""

    # ------------------------------------------------------------------
    # Cost details (new — recommended)
    # ------------------------------------------------------------------

    cost_details: Optional[Dict[str, float]] = None
    """Detailed cost breakdown per metric in USD.
    e.g. {"input": 0.001, "output": 0.002, "total": 0.003}
    Source: Observation.costDetails, ObservationV2.costDetails,
    ClickHouse(cost_details)"""

    provided_cost_details: Optional[Dict[str, float]] = None
    """User-provided cost details (before Langfuse calculation).
    Source: ClickHouse(provided_cost_details)"""

    total_cost: Optional[float] = None
    """Aggregate total cost in USD.
    Source: ObservationV2.totalCost, ClickHouse(total_cost)"""

    # ------------------------------------------------------------------
    # Legacy usage (deprecated — use usage_details / cost_details)
    # ------------------------------------------------------------------

    usage: Optional[Usage] = None
    """Deprecated legacy usage object.
    Source: Observation.usage"""

    prompt_tokens: Optional[int] = None
    """Deprecated. Number of prompt/input tokens.
    Source: Prisma(prompt_tokens)"""

    completion_tokens: Optional[int] = None
    """Deprecated. Number of completion/output tokens.
    Source: Prisma(completion_tokens)"""

    total_tokens: Optional[int] = None
    """Deprecated. Total tokens (prompt + completion).
    Source: Prisma(total_tokens)"""

    unit: Optional[str] = None
    """Deprecated. Unit of measurement for token counts.
    Source: Prisma(unit)"""

    # ------------------------------------------------------------------
    # Legacy cost fields (deprecated — use cost_details)
    # ------------------------------------------------------------------

    input_cost: Optional[float] = None
    """Deprecated. User-provided input cost in USD.
    Source: Prisma(input_cost)"""

    output_cost: Optional[float] = None
    """Deprecated. User-provided output cost in USD.
    Source: Prisma(output_cost)"""

    total_cost_legacy: Optional[float] = None
    """Deprecated. User-provided total cost in USD.
    Source: Prisma(total_cost) — the legacy Decimal column"""

    calculated_input_cost: Optional[float] = None
    """Deprecated. Langfuse-calculated input cost in USD.
    Source: ObservationsView.calculatedInputCost,
    Prisma(calculated_input_cost)"""

    calculated_output_cost: Optional[float] = None
    """Deprecated. Langfuse-calculated output cost in USD.
    Source: ObservationsView.calculatedOutputCost,
    Prisma(calculated_output_cost)"""

    calculated_total_cost: Optional[float] = None
    """Deprecated. Langfuse-calculated total cost in USD.
    Source: ObservationsView.calculatedTotalCost,
    Prisma(calculated_total_cost)"""

    # ------------------------------------------------------------------
    # Pricing (ObservationsView)
    # ------------------------------------------------------------------

    input_price: Optional[float] = None
    """Input price per unit in USD (from matched model pricing).
    Source: ObservationsView.inputPrice"""

    output_price: Optional[float] = None
    """Output price per unit in USD (from matched model pricing).
    Source: ObservationsView.outputPrice"""

    total_price: Optional[float] = None
    """Total price in USD (from matched model pricing).
    Source: ObservationsView.totalPrice"""

    # ------------------------------------------------------------------
    # Pricing tier (ClickHouse migration 0031)
    # ------------------------------------------------------------------

    usage_pricing_tier_id: Optional[str] = None
    """Matched pricing tier ID.
    Source: ClickHouse(usage_pricing_tier_id)"""

    usage_pricing_tier_name: Optional[str] = None
    """Matched pricing tier name.
    Source: ClickHouse(usage_pricing_tier_name)"""

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    prompt_id: Optional[str] = None
    """ID of the linked prompt template.
    Source: Observation.promptId, ObservationV2.promptId,
    Prisma(prompt_id), ClickHouse(prompt_id)"""

    prompt_name: Optional[str] = None
    """Name of the linked prompt template.
    Source: ObservationsView.promptName, ObservationV2.promptName,
    ClickHouse(prompt_name)"""

    prompt_version: Optional[int] = None
    """Version of the linked prompt template.
    Source: ObservationsView.promptVersion, ObservationV2.promptVersion,
    ClickHouse(prompt_version)"""

    # ------------------------------------------------------------------
    # Tool calls (ClickHouse migration 0033)
    # ------------------------------------------------------------------

    tool_definitions: Optional[Dict[str, str]] = None
    """Tool schema definitions for function-calling observations.
    Source: ClickHouse(tool_definitions) — Map(String, String)"""

    tool_calls: Optional[List[str]] = None
    """Serialised tool call objects.
    Source: ClickHouse(tool_calls) — Array(String)"""

    tool_call_names: Optional[List[str]] = None
    """Names of tools that were called.
    Source: ClickHouse(tool_call_names) — Array(String)"""

    # ------------------------------------------------------------------
    # Versioning / deployment context
    # ------------------------------------------------------------------

    version: Optional[str] = None
    """Version identifier for the code or component.
    Source: Observation.version, ObservationV2.version,
    Prisma(version), ClickHouse(version)"""

    environment: Optional[str] = None
    """Tracing environment (e.g. "default", "production", "staging").
    Source: Observation.environment, ObservationV2.environment,
    ClickHouse(environment)"""

    release: Optional[str] = None
    """Release identifier for the application.
    Source: SDK LangfuseOtelSpanAttributes.RELEASE,
    set via LangfuseObservationWrapper.__init__"""

    # ------------------------------------------------------------------
    # Trace-level context (denormalised from parent trace)
    # ------------------------------------------------------------------

    user_id: Optional[str] = None
    """User ID from the parent trace.
    Source: ObservationV2.userId"""

    session_id: Optional[str] = None
    """Session ID from the parent trace.
    Source: ObservationV2.sessionId"""

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------

    bookmarked: Optional[bool] = None
    """Whether the observation is bookmarked in the UI.
    Source: ObservationV2.bookmarked"""

    public: Optional[bool] = None
    """Whether the observation / trace is publicly accessible via URL.
    Source: ObservationV2.public"""

    # ------------------------------------------------------------------
    # Timestamps (server-managed)
    # ------------------------------------------------------------------

    created_at: Optional[datetime] = None
    """When the record was first created in Langfuse.
    Source: ObservationV2.createdAt, Prisma(created_at),
    ClickHouse(created_at)"""

    updated_at: Optional[datetime] = None
    """When the record was last updated in Langfuse.
    Source: ObservationV2.updatedAt, Prisma(updated_at),
    ClickHouse(updated_at)"""

    # ------------------------------------------------------------------
    # Internal / server-only fields
    # ------------------------------------------------------------------

    event_ts: Optional[datetime] = None
    """Internal event timestamp used by ClickHouse ReplacingMergeTree
    for deduplication.
    Source: ClickHouse(event_ts)"""

    is_deleted: Optional[bool] = None
    """Soft-delete flag used by ClickHouse ReplacingMergeTree.
    Source: ClickHouse(is_deleted)"""
