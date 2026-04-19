from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from config import (
    ConfigurationError,
    build_openai_client,
    get_llm_request_options,
    get_llm_token_budgets,
)

logger = logging.getLogger(__name__)

DECISION_PATTERN = re.compile(r"^\s*(YES|NO)\b[:\-\s]*(.*)$", re.IGNORECASE | re.DOTALL)


class BrainError(RuntimeError):
    """Raised when slice evaluation fails."""


class BrainAnalysis(BaseModel):
    """Structured output from the reasoning engine."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    root_cause_determinable: bool
    should_query_greptile: bool
    reason: str = Field(..., min_length=1)
    raw_response: str = Field(..., min_length=1)


class Brain:
    """LLM-backed decision maker for whether local code is sufficient."""

    def __init__(
        self,
        client: OpenAI | Any | None = None,
        model: str | None = None,
        request_options: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if client is None:
            client, default_model = build_openai_client()
            brain_max_tokens, _ = get_llm_token_budgets()
            self._client = client
            self._model = model or default_model
            self._request_options = request_options or get_llm_request_options()
            self._max_tokens = max_tokens or brain_max_tokens
        else:
            self._client = client
            self._model = model or "z-ai/glm4.7"
            self._request_options = request_options or {}
            self._max_tokens = max_tokens or 4096

    def evaluate_slice(
        self,
        error_context: dict[str, Any],
        code_slice: str,
    ) -> BrainAnalysis:
        """Ask the LLM whether the root cause is fully contained in the current slice."""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=self._max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Agent-X's reasoning engine. "
                            "Decide whether the root cause of the reported error can be "
                            "determined entirely from the provided code snippet. "
                            "Don't just find the error. Identify the Category of Underperformance. Is it a Schema Gap, a Logic Loop, or State Drift?"
                            "Ensure the agent maps the exact library from the imports"
                            "Answer with exactly one of these formats:\n"
                            "YES: <brief reason>\n"
                            "NO: <brief reason>\n"
                            "Choose NO whenever upstream callers, external dependencies, "
                            "missing definitions, repository structure, configuration, or "
                            "cross-file context are needed to explain the bug."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Error context:\n"
                            f"{json.dumps(error_context, indent=2, sort_keys=True)}\n\n"
                            "Code slice:\n"
                            f"```python\n{code_slice}\n```"
                        ),
                    },
                ],
                **self._request_options,
            )
            raw_response = self._extract_message_content(response)
            decision, reason = self._parse_decision(raw_response)
            should_query_greptile = decision == "NO"
            return BrainAnalysis(
                root_cause_determinable=not should_query_greptile,
                should_query_greptile=should_query_greptile,
                reason=reason,
                raw_response=raw_response,
            )
        except ConfigurationError:
            logger.exception("LLM configuration is invalid for Brain evaluation.")
            raise
        except OpenAIError as exc:
            logger.exception("LLM evaluation failed while analyzing the current code slice.")
            raise BrainError("Brain evaluation failed while calling the LLM.") from exc
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            logger.exception("Brain could not parse the LLM response.")
            raise BrainError("Brain evaluation returned an unreadable decision.") from exc

    def _extract_message_content(self, response: Any) -> str:
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response content was empty.")
        return content.strip()

    def _parse_decision(self, raw_response: str) -> tuple[str, str]:
        match = DECISION_PATTERN.match(raw_response)
        if match:
            decision = match.group(1).upper()
            reason = match.group(2).strip() or "No reason provided."
            return decision, reason

        normalized = raw_response.strip().upper()
        if normalized.startswith("YES"):
            return "YES", raw_response[3:].lstrip(": -\n\t") or "No reason provided."
        if normalized.startswith("NO"):
            return "NO", raw_response[2:].lstrip(": -\n\t") or "No reason provided."

        raise ValueError("LLM response did not start with YES or NO.")
