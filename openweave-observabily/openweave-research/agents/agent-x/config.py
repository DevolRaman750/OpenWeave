from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


class LLMSettings(BaseModel):
    """Environment-backed settings for the LLM client used by Agent-X."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: Literal["openai", "openrouter", "nvidia"] = "openai"
    api_key: str = Field(..., min_length=1)
    model: str = Field(default="gpt-4.1-mini", min_length=1)
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    http_referer: str | None = None
    app_title: str | None = None
    brain_max_tokens: int = Field(default=4096, ge=256)
    packager_max_tokens: int = Field(default=16384, ge=1024)


class GreptileSettings(BaseModel):
    """Environment-backed settings for Greptile fallback queries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    api_key: str = Field(..., min_length=1)


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    nvidia_api_key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("NVAPI_KEY")
        or os.getenv("GLM_KEY")
    )
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if nvidia_api_key:
        provider = "nvidia"
    elif openrouter_api_key:
        provider = "openrouter"
    else:
        provider = "openai"

    api_key = nvidia_api_key or openrouter_api_key or openai_api_key
    if not api_key:
        raise ConfigurationError(
            "Missing LLM credentials. Set NVIDIA_API_KEY, NVAPI_KEY, GLM_KEY, "
            "OPENROUTER_API_KEY, OPENAI_API_KEY, or LLM_API_KEY in the environment."
        )

    try:
        return LLMSettings.model_validate(
            {
                "provider": provider,
                "api_key": api_key,
                "model": (
                    os.getenv("NVIDIA_MODEL")
                    or os.getenv("GLM_MODEL")
                    or os.getenv("OPENROUTER_MODEL")
                    or os.getenv("OPENAI_MODEL")
                    or (
                        "z-ai/glm4.7"
                        if provider == "nvidia"
                        else (
                            "qwen/qwen3-coder:free"
                            if provider == "openrouter"
                            else "gpt-4.1-mini"
                        )
                    )
                ),
                "base_url": (
                    os.getenv("NVIDIA_BASE_URL")
                    or (
                        "https://integrate.api.nvidia.com/v1"
                        if provider == "nvidia"
                        else None
                    )
                    or os.getenv("OPENROUTER_BASE_URL")
                    or (
                        "https://openrouter.ai/api/v1"
                        if provider == "openrouter"
                        else None
                    )
                    or os.getenv("OPENAI_BASE_URL")
                ),
                "organization": os.getenv("OPENAI_ORG_ID"),
                "project": os.getenv("OPENAI_PROJECT_ID"),
                "http_referer": os.getenv("OPENROUTER_HTTP_REFERER")
                or os.getenv("OPENROUTER_SITE_URL"),
                "app_title": os.getenv("OPENROUTER_TITLE")
                or os.getenv("OPENROUTER_SITE_NAME"),
                "brain_max_tokens": int(os.getenv("LLM_BRAIN_MAX_TOKENS", "4096")),
                "packager_max_tokens": int(os.getenv("LLM_PACKAGER_MAX_TOKENS", "16384")),
            }
        )
    except ValidationError as exc:
        raise ConfigurationError("Invalid LLM configuration in environment variables.") from exc


def build_openai_client() -> tuple[OpenAI, str]:
    settings = get_llm_settings()
    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        organization=settings.organization,
        project=settings.project,
    )
    return client, settings.model


def get_llm_request_options() -> dict[str, Any]:
    settings = get_llm_settings()
    request_options: dict[str, Any] = {}

    if settings.provider == "openrouter":
        extra_headers: dict[str, str] = {}
        if settings.http_referer:
            extra_headers["HTTP-Referer"] = settings.http_referer
        if settings.app_title:
            extra_headers["X-OpenRouter-Title"] = settings.app_title
        if extra_headers:
            request_options["extra_headers"] = extra_headers
        request_options["extra_body"] = {}

    return request_options


def get_llm_token_budgets() -> tuple[int, int]:
    settings = get_llm_settings()
    return settings.brain_max_tokens, settings.packager_max_tokens


@lru_cache(maxsize=1)
def get_greptile_settings() -> GreptileSettings:
    api_key = os.getenv("GREPTILE_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "Missing Greptile credentials. Set GREPTILE_API_KEY in the environment."
        )

    try:
        return GreptileSettings.model_validate({"api_key": api_key})
    except ValidationError as exc:
        raise ConfigurationError(
            "Invalid Greptile configuration in environment variables."
        ) from exc
