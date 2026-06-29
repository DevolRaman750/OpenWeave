"""
judge.py — Claude Sonnet judge model wrapping DeepEvalBaseLLM.

Why this exists
---------------
DeepEval's built-in model providers default to OpenAI. We're using Claude
as the judge, so we wrap Anthropic's SDK in DeepEval's
``DeepEvalBaseLLM`` abstract interface. Any DeepEval metric that accepts a
``model=`` parameter can then use Claude transparently.

Schema-driven generation
------------------------
Recent DeepEval metrics call ``a_generate(prompt, schema=PydanticModel)``
expecting a structured response. Claude doesn't natively return JSON
matching a Pydantic schema, so we:
    1. Append the JSON schema to the prompt with explicit instructions.
    2. Extract the first JSON object from the response.
    3. Validate against the schema; raise on mismatch.

Environment
-----------
    OPENWEAVE_JUDGE_MODEL   — model id (default: claude-sonnet-4-6)
    ANTHROPIC_API_KEY       — required for live calls
    OPENWEAVE_JUDGE_MAX_TOKENS — max output tokens (default: 2048)
    OPENWEAVE_JUDGE_TEMP    — temperature (default: 0.0 — deterministic)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Type

try:
    import anthropic
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "anthropic package not installed; run `pip install anthropic`"
    ) from exc

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "deepeval package not installed; run `pip install deepeval`"
    ) from exc


DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

# External-call resilience for the judge. LLM generation legitimately takes
# longer than an embedding, so the timeout is larger; retries are bounded and
# the SDK applies exponential backoff between them.
JUDGE_TIMEOUT_SECONDS: float = float(os.environ.get("OPENWEAVE_JUDGE_TIMEOUT", "60"))
JUDGE_MAX_RETRIES: int = int(os.environ.get("OPENWEAVE_JUDGE_MAX_RETRIES", "3"))


def _get_env_model() -> str:
    return os.environ.get("OPENWEAVE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def _get_env_max_tokens() -> int:
    return int(os.environ.get("OPENWEAVE_JUDGE_MAX_TOKENS", "2048"))


def _get_env_temperature() -> float:
    return float(os.environ.get("OPENWEAVE_JUDGE_TEMP", "0.0"))


class ClaudeJudge(DeepEvalBaseLLM):
    """A DeepEval-compatible judge model backed by Anthropic's Claude API."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model_name = model or _get_env_model()
        self._max_tokens = max_tokens if max_tokens is not None else _get_env_max_tokens()
        self._temperature = (
            temperature if temperature is not None else _get_env_temperature()
        )
        self._api_key = api_key  # None -> anthropic SDK reads ANTHROPIC_API_KEY
        self._sync_client: Optional[anthropic.Anthropic] = None
        self._async_client: Optional[anthropic.AsyncAnthropic] = None

    # ------------------------------------------------------------------
    # DeepEvalBaseLLM interface
    # ------------------------------------------------------------------

    def load_model(self) -> Any:
        if self._sync_client is None:
            self._sync_client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=JUDGE_TIMEOUT_SECONDS,
                max_retries=JUDGE_MAX_RETRIES,
            )
        return self._sync_client

    def get_model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, schema: Optional[Type[Any]] = None) -> Any:
        client = self.load_model()
        full_prompt = _augment_prompt_for_schema(prompt, schema)
        message = client.messages.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = _extract_text(message)
        if schema is None:
            return text
        return _parse_to_schema(text, schema)

    async def a_generate(
        self, prompt: str, schema: Optional[Type[Any]] = None
    ) -> Any:
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=JUDGE_TIMEOUT_SECONDS,
                max_retries=JUDGE_MAX_RETRIES,
            )
        full_prompt = _augment_prompt_for_schema(prompt, schema)
        message = await self._async_client.messages.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = _extract_text(message)
        if schema is None:
            return text
        return _parse_to_schema(text, schema)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(message: Any) -> str:
    """Pull the first text block out of an Anthropic message response."""
    blocks = getattr(message, "content", None) or []
    for block in blocks:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
        # fallback when content blocks come back as dicts
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "") or ""
    return ""


def _augment_prompt_for_schema(prompt: str, schema: Optional[Type[Any]]) -> str:
    if schema is None:
        return prompt
    try:
        schema_json = schema.model_json_schema()
    except AttributeError:
        # Older Pydantic v1 fallback.
        schema_json = schema.schema()  # type: ignore[attr-defined]
    return (
        f"{prompt}\n\n"
        "You MUST respond with a single JSON object that strictly conforms to "
        "the following JSON schema. Do not include any prose, code fences, or "
        "commentary — just the raw JSON object.\n\n"
        f"JSON Schema:\n{json.dumps(schema_json, indent=2)}\n"
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_to_schema(text: str, schema: Type[Any]) -> Any:
    """Extract the first JSON object in ``text`` and validate against schema."""
    cleaned = _strip_code_fences(text).strip()
    candidates = [cleaned]
    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        candidates.append(match.group(0))

    last_err: Optional[Exception] = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        try:
            return schema.model_validate(data)
        except AttributeError:
            return schema.parse_obj(data)  # type: ignore[attr-defined]
        except Exception as e:
            last_err = e
    raise ValueError(
        f"Could not parse judge response into {schema.__name__}: "
        f"{last_err}; raw text: {text[:300]}"
    )


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text)
