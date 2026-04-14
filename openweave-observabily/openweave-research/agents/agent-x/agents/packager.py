from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from agents.brain import BrainAnalysis
from config import (
    ConfigurationError,
    build_openai_client,
    get_llm_request_options,
    get_llm_token_budgets,
)
from schemas.acm_manifest import CompatibilityManifest
from schemas.input_packet import TraceInput

logger = logging.getLogger(__name__)

JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class PackagerError(RuntimeError):
    """Raised when ACM generation fails."""


class Packager:
    """LLM-backed ACM builder with schema validation."""

    def __init__(
        self,
        client: OpenAI | Any | None = None,
        model: str | None = None,
        request_options: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if client is None:
            client, default_model = build_openai_client()
            _, packager_max_tokens = get_llm_token_budgets()
            self._client = client
            self._model = model or default_model
            self._request_options = request_options or get_llm_request_options()
            self._max_tokens = max_tokens or packager_max_tokens
        else:
            self._client = client
            self._model = model or "z-ai/glm4.7"
            self._request_options = request_options or {}
            self._max_tokens = max_tokens or 16384

    def generate_acm(
        self,
        input_packet: TraceInput | dict[str, Any],
        primary_slice: str,
        detective_context: dict[str, Any] | None,
        brain_analysis: BrainAnalysis | dict[str, Any],
    ) -> CompatibilityManifest:
        """Generate and validate the ACM JSON-LD manifest."""

        try:
            trace_input = self._normalize_input_packet(input_packet)
            normalized_brain = self._normalize_brain_analysis(brain_analysis)
        except ValidationError as exc:
            logger.exception("Packager inputs failed validation.")
            raise PackagerError("Packager inputs did not match the expected schemas.") from exc

        try:
            normalized_detective_context = detective_context or {}
            schema = CompatibilityManifest.model_json_schema(by_alias=True)

            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Agent-X's ACM packaging engine. "
                            "Transform the provided debugging context into a single valid "
                            "JSON-LD object that conforms exactly to the supplied schema. "
                            "Output JSON only with no markdown. "
                            "Use '@context' and '@type' exactly as defined. "
                            "Ground all fields in the provided materials. "
                            "You may infer search queries, architecture pattern names, and "
                            "technical constraints when they are strongly supported by the "
                            "error, code, dependency evidence, or semantic context. "
                            "Do not invent precise package versions unless explicit evidence "
                            "is present. If a runtime or architecture detail is uncertain, use "
                            "a conservative but non-empty description."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Target schema:\n{json.dumps(schema, indent=2)}\n\n"
                            f"Trace input:\n{json.dumps(trace_input.model_dump(), indent=2)}\n\n"
                            f"Primary slice:\n```python\n{primary_slice}\n```\n\n"
                            "Detective context:\n"
                            f"{json.dumps(normalized_detective_context, indent=2, default=str)}\n\n"
                            "Brain analysis:\n"
                            f"{json.dumps(normalized_brain.model_dump(), indent=2)}\n\n"
                            "Return one JSON object only. "
                            "Populate the ACM so it captures the failing node, relevant code "
                            "context, dependencies, architecture pattern, search queries, and "
                            "technical constraints needed for downstream remediation search."
                        ),
                    },
                ],
                **self._request_options,
            )
            llm_output = self._extract_message_content(response)
            llm_output = self._extract_json_text(llm_output)
            return CompatibilityManifest.model_validate_json(llm_output)
        except ConfigurationError:
            logger.exception("LLM configuration is invalid for ACM generation.")
            raise
        except ValidationError as exc:
            logger.exception("ACM output failed schema validation.")
            raise PackagerError("Generated ACM did not match the CompatibilityManifest schema.") from exc
        except OpenAIError as exc:
            logger.exception("LLM ACM generation failed.")
            raise PackagerError("ACM generation failed while calling the LLM.") from exc
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            logger.exception("Packager could not parse the LLM ACM response.")
            raise PackagerError("ACM generation returned unreadable output.") from exc

    def _normalize_input_packet(
        self,
        input_packet: TraceInput | dict[str, Any],
    ) -> TraceInput:
        if isinstance(input_packet, TraceInput):
            return input_packet
        return TraceInput.model_validate(input_packet)

    def _normalize_brain_analysis(
        self,
        brain_analysis: BrainAnalysis | dict[str, Any],
    ) -> BrainAnalysis:
        if isinstance(brain_analysis, BrainAnalysis):
            return brain_analysis
        return BrainAnalysis.model_validate(brain_analysis)

    def _extract_message_content(self, response: Any) -> str:
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response content was empty.")
        return content.strip()

    def _extract_json_text(self, raw_output: str) -> str:
        match = JSON_BLOCK_PATTERN.search(raw_output)
        if match:
            return match.group(1).strip()
        return raw_output.strip()
