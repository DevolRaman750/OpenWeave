from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictSchema(BaseModel):
    """Base schema with strict validation and no undeclared fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class ProblemDomain(StrictSchema):
    error_type: str = Field(..., min_length=1)
    symptom: str = Field(..., min_length=1)
    failing_node: str = Field(..., min_length=1)


class CodebaseContext(StrictSchema):
    primary_slice: str = Field(..., min_length=1)
    filepath: str = Field(..., min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    architecture_pattern: str = Field(..., min_length=1)

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        if any(not dependency.strip() for dependency in value):
            raise ValueError("dependencies must not contain empty values")
        return value


class TechnicalConstraints(StrictSchema):
    runtime: str = Field(..., min_length=1)
    required_libraries: list[str] = Field(default_factory=list)

    @field_validator("required_libraries")
    @classmethod
    def validate_required_libraries(cls, value: list[str]) -> list[str]:
        if any(not library.strip() for library in value):
            raise ValueError("required_libraries must not contain empty values")
        return value


class SearchIntent(StrictSchema):
    queries: list[str] = Field(..., min_length=1)
    repomaster_query: str | None = None
    technical_constraints: TechnicalConstraints

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, value: list[str]) -> list[str]:
        if any(not query.strip() for query in value):
            raise ValueError("queries must not contain empty values")
        return value

    @field_validator("repomaster_query")
    @classmethod
    def validate_repomaster_query(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("repomaster_query must not be empty")
        return value


class CompatibilityManifest(StrictSchema):
    context: Literal["https://openweave.ai/schemas/acm/v1"] = Field(
        default="https://openweave.ai/schemas/acm/v1",
        alias="@context",
    )
    type: Literal["CompatibilityManifest"] = Field(
        default="CompatibilityManifest",
        alias="@type",
    )
    problem_domain: ProblemDomain
    codebase_context: CodebaseContext
    search_intent: SearchIntent
