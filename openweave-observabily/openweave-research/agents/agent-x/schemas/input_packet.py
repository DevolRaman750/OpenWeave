from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    commit_id: str = Field(..., min_length=7)
    filepath: str = Field(..., min_length=1)
    lineno: int = Field(..., gt=0)
    function: str = Field(..., min_length=1)
    error_context: dict[str, Any] = Field(..., min_length=1)

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
        if len(commit) < 7:
            raise ValueError("commit_id must be at least 7 characters long")
        return commit

    @field_validator("error_context")
    @classmethod
    def validate_error_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("error_context must not be empty")
        if any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError("error_context keys must be non-empty strings")
        return value
