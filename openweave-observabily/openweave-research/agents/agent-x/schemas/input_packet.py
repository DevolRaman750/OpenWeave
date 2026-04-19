from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TraceInput(BaseModel):
    """Validated coordinate packet for Agent-X trace ingestion."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    trace_id: str = Field(..., min_length=1)
    github_pat: str = Field(..., min_length=1)
    repository: str = Field(..., min_length=3)
    commit_id: str = Field(..., min_length=1)
    filepath: str = Field(..., min_length=1)
    lineno: int = Field(..., gt=0)
    function: str = Field(..., min_length=1)
    error_context: dict[str, Any] = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_ingestion_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        code_location = normalized.get("code_location")
        if isinstance(code_location, dict):
            normalized.setdefault("filepath", code_location.get("filepath"))
            normalized.setdefault("lineno", code_location.get("lineno"))
            normalized.setdefault("function", code_location.get("function"))
            normalized.pop("code_location", None)

        repository = normalized.get("repository")
        if isinstance(repository, str):
            repo = repository.strip()
            if repo.startswith("http://") or repo.startswith("https://"):
                parsed = urlparse(repo)
                if parsed.netloc.lower() == "github.com":
                    parts = [part for part in parsed.path.split("/") if part]
                    if len(parts) >= 2:
                        normalized["repository"] = f"{parts[0]}/{parts[1]}"

        return normalized

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        parts = value.split("/")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError("repository must be in 'owner/repo' format")
        return value

    @field_validator("filepath", "function", "trace_id", "github_pat")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be empty")
        return value

    @field_validator("commit_id")
    @classmethod
    def validate_commit_id(cls, value: str) -> str:
        commit = value.strip()
        if not commit:
            raise ValueError("commit_id must not be empty")
        return commit

    @field_validator("error_context")
    @classmethod
    def validate_error_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("error_context must not be empty")
        if any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError("error_context keys must be non-empty strings")
        return value
