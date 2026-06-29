"""
case_builder.py — Convert an EvaluationTask.context dict to an LLMTestCase.

The upstream incident_classification layer pre-builds a per-category
context bag. This module unpacks that bag into the input/actual_output/
retrieval_context fields DeepEval's ``LLMTestCase`` expects, so each
category gets only the fields its metric actually reads.

Mapping (one source of truth)
-----------------------------
    PROMPT          : input  = trace_input + suspicious_payloads
                      output = trace_output
    RAG             : input  = trace_input
                      output = trace_output
                      retrieval_context = list of retrieval span outputs
    TOOL_INVOCATION : input  = primary_span_input  (or trace_input)
                      output = primary_span_output (or trace_output) +
                               summarised tool calls (args + outputs)
    LLM_GENERATION  : input  = primary_span_input  (or trace_input)
                      output = primary_span_output (or trace_output)
"""

from __future__ import annotations

import json
from typing import Any

try:
    from deepeval.test_case import LLMTestCase
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "deepeval package not installed; run `pip install deepeval`"
    ) from exc

from openweave_core.deep_evaluation.contracts import (
    EvaluationCategory,
    EvaluationTask,
)


# Hard cap individual text fields so a runaway trace can't bomb a judge.
_MAX_TEXT_CHARS = 16_000


def build_test_case(task: EvaluationTask) -> LLMTestCase:
    """Dispatch on category to build the right LLMTestCase shape."""
    ctx = task.context or {}
    if task.category == EvaluationCategory.PROMPT:
        return _build_prompt_case(ctx)
    if task.category == EvaluationCategory.RAG:
        return _build_rag_case(ctx)
    if task.category == EvaluationCategory.TOOL_INVOCATION:
        return _build_tool_case(ctx)
    if task.category == EvaluationCategory.LLM_GENERATION:
        return _build_generation_case(ctx)
    # OBSERVABILITY / SAFETY fall through to a minimal case — runner
    # filters these out before they get here, but be defensive.
    return _build_generation_case(ctx)


# ---------------------------------------------------------------------------
# Per-category builders
# ---------------------------------------------------------------------------

def _build_prompt_case(ctx: dict[str, Any]) -> LLMTestCase:
    suspicious = ctx.get("suspicious_payloads") or []
    trace_input = _str(ctx.get("trace_input"))
    extra = ""
    if suspicious:
        joined = "\n---\n".join(_str(t) for t in suspicious[:5])
        extra = f"\n\n[Suspicious payloads observed during this trace]\n{joined}"
    return LLMTestCase(
        input=_truncate(trace_input + extra),
        actual_output=_truncate(_str(ctx.get("trace_output"))),
    )


def _build_rag_case(ctx: dict[str, Any]) -> LLMTestCase:
    retrieval = ctx.get("retrieval_context") or []
    context_chunks: list[str] = []
    for item in retrieval:
        if isinstance(item, dict):
            payload = item.get("output") or item.get("input") or ""
            label = item.get("tool_name") or item.get("span_id") or "retrieval"
            chunk = f"[{label}]\n{_str(payload)}"
        else:
            chunk = _str(item)
        if chunk:
            context_chunks.append(_truncate(chunk, _MAX_TEXT_CHARS // 4))
    if not context_chunks:
        context_chunks = [""]  # FaithfulnessMetric requires non-empty list
    return LLMTestCase(
        input=_truncate(_str(ctx.get("trace_input"))),
        actual_output=_truncate(_str(ctx.get("trace_output"))),
        retrieval_context=context_chunks,
    )


def _build_tool_case(ctx: dict[str, Any]) -> LLMTestCase:
    primary_in = _str(ctx.get("primary_span_input")) or _str(ctx.get("trace_input"))
    primary_out = _str(ctx.get("primary_span_output")) or _str(ctx.get("trace_output"))
    tool_calls = ctx.get("tool_calls") or []
    tool_block = ""
    if tool_calls:
        summary = []
        for tc in tool_calls[:5]:
            if not isinstance(tc, dict):
                continue
            summary.append(
                f"- tool={tc.get('tool_name')}  "
                f"args={_truncate(_str(tc.get('input')), 500)}  "
                f"output={_truncate(_str(tc.get('output')), 500)}"
            )
        if summary:
            tool_block = "\n\n[Tool calls observed]\n" + "\n".join(summary)
    return LLMTestCase(
        input=_truncate(primary_in),
        actual_output=_truncate(primary_out + tool_block),
    )


def _build_generation_case(ctx: dict[str, Any]) -> LLMTestCase:
    return LLMTestCase(
        input=_truncate(
            _str(ctx.get("primary_span_input")) or _str(ctx.get("trace_input"))
        ),
        actual_output=_truncate(
            _str(ctx.get("primary_span_output")) or _str(ctx.get("trace_output"))
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _truncate(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"
