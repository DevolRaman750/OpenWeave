import ast
import copy
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


RUNTIME_ALIASES = {
    "python": "python-3.x",
    "python3": "python-3.x",
    "python3x": "python-3.x",
    "python 3": "python-3.x",
    "python3.x": "python-3.x",
    "python 3.x": "python-3.x",
    "py3": "python-3.x",
    "node": "node-18+",
    "nodejs": "node-18+",
    "node.js": "node-18+",
    "javascript": "node-18+",
    "typescript": "node-18+",
    "java": "java-17+",
    "go": "go-1.22+",
    "golang": "go-1.22+",
}

LANGUAGE_ALIASES = {
    "python": "python",
    "py": "python",
    "python3": "python",
    "javascript": "javascript",
    "js": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "node.js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "java": "java",
    "go": "go",
    "golang": "go",
    "csharp": "csharp",
    "c#": "csharp",
    "dotnet": "csharp",
}

DEFAULT_RUNTIME_BY_LANGUAGE = {
    "python": "python-3.x",
    "javascript": "node-18+",
    "typescript": "node-18+",
    "java": "java-17+",
    "go": "go-1.22+",
    "csharp": "dotnet-8",
}

ERROR_TYPE_ALIASES = {
    "timeoutwarning": "TimeoutWarning",
    "timeoutexceeded": "TimeoutWarning",
    "timeouterror": "TimeoutError",
    "timeout": "TimeoutWarning",
    "keyerror": "KeyError",
    "valueerror": "ValueError",
    "indexerror": "IndexError",
    "attributeerror": "AttributeError",
    "typeerror": "TypeError",
    "connectionerror": "ConnectionError",
    "http502": "BadGatewayError",
    "badgateway": "BadGatewayError",
    "gatewayerror": "BadGatewayError",
    "ratelimit": "RateLimitError",
    "toomanyrequests": "RateLimitError",
}

ARCHITECTURE_PATTERN_ALIASES = {
    "stategraphwithtoolcalling": "State Graph with Tool Calling",
    "stategraph_tool_calling": "State Graph with Tool Calling",
    "stategraphwithtools": "State Graph with Tool Calling",
    "langgraphstategraphwithtoolcalling": "State Graph with Tool Calling",
    "reactagent": "ReAct Agent",
    "multiagent": "Multi-Agent Orchestration",
    "multiagentorchestration": "Multi-Agent Orchestration",
    "retrievalaugmentedgeneration": "RAG",
    "rag": "RAG",
}

LIBRARY_ALIAS_MAP = {
    "langgraph": "langgraph",
    "langchain": "langchain",
    "langchain-core": "langchain-core",
    "langchain_core": "langchain-core",
    "autogen": "autogen",
    "openai": "openai",
    "anthropic": "anthropic",
    "azure-openai": "azure-openai",
    "azureopenai": "azure-openai",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "pydantic": "pydantic",
    "requests": "requests",
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
    "transformers": "transformers",
}

STOP_TOKENS = {
    "the", "and", "for", "with", "from", "this", "that", "into", "using", "used", "use", "fix",
    "runtime", "libraries", "library", "error", "warning", "node", "execution", "seconds", "state",
}

STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", set()))


class ACMValidationError(ValueError):
    """Raised when ACM payload is invalid for Stage 1 intake."""


@dataclass(slots=True)
class Stage1IntakeResult:
    canonical_manifest: dict[str, Any]
    inferred_fields: dict[str, Any]
    warnings: list[str]
    compatibility_signals: dict[str, Any]
    query_pack: dict[str, Any]
    scheduler_context_brief: dict[str, Any]


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _as_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ACMValidationError(f"Missing or invalid required string field: {field_name}")
    return _normalize_whitespace(value)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(_normalize_whitespace(item))
        return result
    return []


def _normalize_runtime(runtime_raw: str) -> str:
    if not runtime_raw:
        return ""
    key = _norm_key(runtime_raw)
    return RUNTIME_ALIASES.get(key, _normalize_whitespace(runtime_raw).lower())


def _normalize_language(language_raw: str) -> str:
    if not language_raw:
        return ""
    key = _norm_key(language_raw)
    return LANGUAGE_ALIASES.get(key, _normalize_whitespace(language_raw).lower())


def _infer_language_from_text(*texts: str) -> str:
    corpus = " ".join(t for t in texts if isinstance(t, str)).lower()
    hints = {
        "python": ["python", "py"],
        "javascript": ["javascript", "node.js", "nodejs", " js "],
        "typescript": ["typescript", " ts "],
        "java": [" java "],
        "go": [" golang ", " go "],
    }
    wrapped = f" {corpus} "
    for language, markers in hints.items():
        for marker in markers:
            if marker in wrapped:
                return language
    return ""


def _normalize_error_type(error_type_raw: str) -> str:
    if not error_type_raw:
        return "UnknownError"
    key = _norm_key(error_type_raw)
    return ERROR_TYPE_ALIASES.get(key, _normalize_whitespace(error_type_raw))


def _normalize_architecture_pattern(pattern_raw: str) -> str:
    if not isinstance(pattern_raw, str) or not pattern_raw.strip():
        return "Unknown Architecture"
    normalized = _normalize_whitespace(pattern_raw)
    key = _norm_key(normalized)
    return ARCHITECTURE_PATTERN_ALIASES.get(key, normalized)


def _extract_import_dependencies(primary_slice: str) -> list[str]:
    inferred: set[str] = set()
    try:
        tree = ast.parse(primary_slice)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        inferred.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    inferred.add(node.module)
    except Exception:
        for match in re.finditer(r"^\s*import\s+([a-zA-Z0-9_\.]+)", primary_slice, re.MULTILINE):
            inferred.add(match.group(1))
        for match in re.finditer(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+", primary_slice, re.MULTILINE):
            inferred.add(match.group(1))
    return sorted(inferred)


def _normalize_dependency(value: str) -> str:
    cleaned = _normalize_whitespace(value)
    if cleaned.startswith("from ") and " import " in cleaned:
        cleaned = cleaned.split(" import ", 1)[0].replace("from ", "", 1).strip()
    if cleaned.startswith("import "):
        cleaned = cleaned.replace("import ", "", 1).strip()
    return cleaned


def _dependency_root(dependency: str) -> str:
    cleaned = dependency.strip()
    if not cleaned:
        return ""
    root = cleaned.split(".", 1)[0].strip()
    return root


def _normalize_library_name(value: str) -> str:
    key = _norm_key(value.replace("_", "-").replace(".", "-"))
    return LIBRARY_ALIAS_MAP.get(key, value.replace("_", "-").strip().lower())


def _extract_candidate_tokens(*texts: str) -> list[str]:
    tokens = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9_\-\.]{1,}", text.lower()))
    return tokens


def _infer_required_libraries(
    dependencies: list[str],
    existing_required_libraries: list[str],
    symptom: str,
    repomaster_query: str,
    queries: list[str],
    primary_slice: str,
) -> tuple[list[str], dict[str, Any]]:
    inferred_sources: dict[str, Any] = {
        "from_dependencies": [],
        "from_tokens": [],
    }
    candidates: set[str] = set()

    for lib in existing_required_libraries:
        normalized = _normalize_library_name(lib)
        if normalized:
            candidates.add(normalized)

    for dep in dependencies:
        dep_root = _dependency_root(dep)
        if not dep_root:
            continue
        if dep_root in STDLIB_MODULES:
            continue
        normalized = _normalize_library_name(dep_root)
        if normalized:
            candidates.add(normalized)
            inferred_sources["from_dependencies"].append(normalized)

    all_queries = " ".join(queries)
    token_candidates = _extract_candidate_tokens(symptom, repomaster_query, all_queries, primary_slice)
    for token in token_candidates:
        normalized = _normalize_library_name(token)
        if normalized in LIBRARY_ALIAS_MAP.values():
            candidates.add(normalized)
            inferred_sources["from_tokens"].append(normalized)

    return sorted(candidates), {
        "from_dependencies": sorted(set(inferred_sources["from_dependencies"])),
        "from_tokens": sorted(set(inferred_sources["from_tokens"])),
    }


def _build_semantic_intent_tokens(
    repomaster_query: str,
    symptom: str,
    error_type: str,
    architecture_pattern: str,
    required_libraries: list[str],
    limit: int = 24,
) -> list[str]:
    corpus_tokens = _extract_candidate_tokens(
        repomaster_query,
        symptom,
        error_type,
        architecture_pattern,
        " ".join(required_libraries),
    )
    filtered = [
        token for token in corpus_tokens
        if token not in STOP_TOKENS and len(token) >= 3
    ]
    counts = Counter(filtered)
    return [token for token, _ in counts.most_common(limit)]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result


def _normalize_query_text(value: str) -> str:
    cleaned = _normalize_whitespace(value)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return cleaned


def _build_query_pack(
    repomaster_query: str,
    seed_queries: list[str],
    symptom: str,
    error_type: str,
    failing_node: str,
    architecture_pattern: str,
    runtime: str,
    language: str,
    required_libraries: list[str],
    max_retrieval_queries: int = 18,
) -> dict[str, Any]:
    normalized_seeds = _dedupe_preserve_order(
        [_normalize_query_text(q) for q in seed_queries if isinstance(q, str) and q.strip()]
    )
    if repomaster_query not in normalized_seeds:
        normalized_seeds.insert(0, _normalize_query_text(repomaster_query))

    libraries_clause = ", ".join(required_libraries[:8]) if required_libraries else ""
    runtime_clause = runtime or ""
    language_clause = language or ""
    failing_node_clause = failing_node or "unknown_node"

    template_queries: list[str] = []
    base_templates = [
        f"fix {error_type} {symptom} in {architecture_pattern}",
        f"resolve {error_type} at {failing_node_clause} for {architecture_pattern}",
        f"{architecture_pattern} {failing_node_clause} timeout mitigation {error_type}",
        f"{error_type} {symptom} call path {failing_node_clause}",
    ]

    for base in base_templates:
        base_norm = _normalize_query_text(base)
        template_queries.append(base_norm)
        if runtime_clause:
            template_queries.append(_normalize_query_text(f"{base_norm} runtime {runtime_clause}"))
        if language_clause:
            template_queries.append(_normalize_query_text(f"{base_norm} language {language_clause}"))
        if libraries_clause:
            template_queries.append(_normalize_query_text(f"{base_norm} libraries {libraries_clause}"))
        if runtime_clause and libraries_clause:
            template_queries.append(_normalize_query_text(f"{base_norm} runtime {runtime_clause} libraries {libraries_clause}"))

    constraint_fragments = []
    if runtime_clause:
        constraint_fragments.append(f"runtime {runtime_clause}")
    if language_clause:
        constraint_fragments.append(f"language {language_clause}")
    if libraries_clause:
        constraint_fragments.append(f"libraries {libraries_clause}")
    constraints_suffix = " ".join(constraint_fragments).strip()

    constraint_injected_queries: list[str] = []
    for seed in normalized_seeds:
        seed_norm = _normalize_query_text(seed)
        if constraints_suffix and constraints_suffix.lower() not in seed_norm.lower():
            constraint_injected_queries.append(_normalize_query_text(f"{seed_norm} {constraints_suffix}"))
        else:
            constraint_injected_queries.append(seed_norm)

    retrieval_queries = _dedupe_preserve_order(
        [_normalize_query_text(repomaster_query)]
        + constraint_injected_queries
        + template_queries
        + normalized_seeds
    )[:max_retrieval_queries]

    return {
        "seed_queries": normalized_seeds,
        "template_queries": _dedupe_preserve_order(template_queries),
        "constraint_injected_queries": _dedupe_preserve_order(constraint_injected_queries),
        "retrieval_queries": retrieval_queries,
    }


def _compact_brief_value(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated-depth]"

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            if index >= 24:
                result["__truncated_keys__"] = len(items) - 24
                break
            result[str(key)] = _compact_brief_value(item, depth + 1)
        return result

    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            if index >= 16:
                result.append(f"[truncated_items:{len(value) - 16}]")
                break
            result.append(_compact_brief_value(item, depth + 1))
        return result

    if isinstance(value, str):
        normalized = _normalize_whitespace(value)
        if len(normalized) > 320:
            return normalized[:320] + " ...[truncated]"
        return normalized

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    return str(value)


def _build_scheduler_context_brief(
    canonical_manifest: dict[str, Any],
    compatibility_signals: dict[str, Any],
    query_pack: dict[str, Any],
) -> dict[str, Any]:
    problem_domain = canonical_manifest.get("problem_domain", {})
    codebase_context = canonical_manifest.get("codebase_context", {})
    search_intent = canonical_manifest.get("search_intent", {})
    technical_constraints = search_intent.get("technical_constraints", {}) if isinstance(search_intent, dict) else {}

    brief = {
        "problem_domain": {
            "error_type": problem_domain.get("error_type", ""),
            "symptom": problem_domain.get("symptom", ""),
            "failing_node": problem_domain.get("failing_node", ""),
        },
        "codebase_context": {
            "filepath": codebase_context.get("filepath", ""),
            "architecture_pattern": codebase_context.get("architecture_pattern", ""),
            "dependencies": codebase_context.get("dependencies", []),
            "primary_slice_excerpt": codebase_context.get("primary_slice", ""),
        },
        "search_intent": {
            "repomaster_query": search_intent.get("repomaster_query", ""),
            "queries": search_intent.get("queries", []),
            "technical_constraints": {
                "runtime": technical_constraints.get("runtime", ""),
                "language": technical_constraints.get("language", ""),
                "required_libraries": technical_constraints.get("required_libraries", []),
            },
        },
        "compatibility_signals": compatibility_signals,
        "query_pack_preview": {
            "retrieval_queries": query_pack.get("retrieval_queries", [])[:8],
            "template_queries": query_pack.get("template_queries", [])[:6],
        },
    }

    return _compact_brief_value(brief)


def normalize_acm_manifest(payload: dict[str, Any]) -> Stage1IntakeResult:
    if not isinstance(payload, dict):
        raise ACMValidationError("ACM payload must be a JSON object")

    required_sections = ("problem_domain", "codebase_context", "search_intent")
    for section in required_sections:
        if section not in payload or not isinstance(payload.get(section), dict):
            raise ACMValidationError(f"Missing or invalid required object field: {section}")

    problem_domain = payload["problem_domain"]
    codebase_context = payload["codebase_context"]
    search_intent = payload["search_intent"]
    technical_constraints = search_intent.get("technical_constraints")
    if technical_constraints is None:
        technical_constraints = {}
    if not isinstance(technical_constraints, dict):
        raise ACMValidationError("search_intent.technical_constraints must be an object when present")

    error_type_raw = _as_non_empty_str(problem_domain.get("error_type"), "problem_domain.error_type")
    symptom = _as_non_empty_str(problem_domain.get("symptom"), "problem_domain.symptom")
    primary_slice = _as_non_empty_str(codebase_context.get("primary_slice"), "codebase_context.primary_slice")
    repomaster_query = _as_non_empty_str(search_intent.get("repomaster_query"), "search_intent.repomaster_query")

    architecture_pattern_raw = _normalize_whitespace(str(codebase_context.get("architecture_pattern", "")).strip())
    failing_node = _normalize_whitespace(str(problem_domain.get("failing_node", "")).strip())
    runtime_raw = _normalize_whitespace(str(technical_constraints.get("runtime", "")).strip())
    language_raw = _normalize_whitespace(str(technical_constraints.get("language", "")).strip())

    dependencies = [_normalize_dependency(dep) for dep in _as_string_list(codebase_context.get("dependencies"))]
    dependencies = sorted({dep for dep in dependencies if dep})
    inferred_fields: dict[str, Any] = {}
    warnings: list[str] = []

    if not dependencies:
        inferred_dependencies = _extract_import_dependencies(primary_slice)
        if inferred_dependencies:
            dependencies = inferred_dependencies
            inferred_fields["dependencies"] = {
                "source": "codebase_context.primary_slice imports",
                "value": inferred_dependencies,
            }
            warnings.append("codebase_context.dependencies was empty; inferred dependencies from primary_slice imports")
        else:
            warnings.append("codebase_context.dependencies was empty and no imports were inferred from primary_slice")

    normalized_error_type = _normalize_error_type(error_type_raw)
    if normalized_error_type != error_type_raw:
        inferred_fields["normalized_error_type"] = {
            "from": error_type_raw,
            "to": normalized_error_type,
        }

    normalized_architecture_pattern = _normalize_architecture_pattern(architecture_pattern_raw)
    if architecture_pattern_raw and normalized_architecture_pattern != architecture_pattern_raw:
        inferred_fields["normalized_architecture_pattern"] = {
            "from": architecture_pattern_raw,
            "to": normalized_architecture_pattern,
        }
    if not architecture_pattern_raw:
        warnings.append("codebase_context.architecture_pattern is missing; set to Unknown Architecture")

    normalized_language = _normalize_language(language_raw)
    if not normalized_language:
        inferred_language = _infer_language_from_text(runtime_raw, repomaster_query, primary_slice)
        if inferred_language:
            normalized_language = inferred_language
            inferred_fields["language"] = {
                "source": "runtime/query/code heuristics",
                "value": normalized_language,
            }

    normalized_runtime = _normalize_runtime(runtime_raw)
    if not normalized_runtime and normalized_language:
        normalized_runtime = DEFAULT_RUNTIME_BY_LANGUAGE.get(normalized_language, "")
        if normalized_runtime:
            inferred_fields["runtime"] = {
                "source": "language-default mapping",
                "value": normalized_runtime,
            }

    if not normalized_runtime:
        warnings.append("technical_constraints.runtime is missing and could not be inferred")
    if not normalized_language:
        warnings.append("technical_constraints.language is missing and could not be inferred")

    queries = _as_string_list(search_intent.get("queries"))
    existing_required_libraries = _as_string_list(technical_constraints.get("required_libraries"))
    required_libraries, required_library_sources = _infer_required_libraries(
        dependencies=dependencies,
        existing_required_libraries=existing_required_libraries,
        symptom=symptom,
        repomaster_query=repomaster_query,
        queries=queries,
        primary_slice=primary_slice,
    )
    if required_libraries and required_libraries != sorted(set(existing_required_libraries)):
        inferred_fields["required_libraries"] = {
            "source": required_library_sources,
            "value": required_libraries,
        }

    semantic_intent_tokens = _build_semantic_intent_tokens(
        repomaster_query=repomaster_query,
        symptom=symptom,
        error_type=normalized_error_type,
        architecture_pattern=normalized_architecture_pattern,
        required_libraries=required_libraries,
    )

    canonical_manifest = copy.deepcopy(payload)
    canonical_manifest.setdefault("problem_domain", {})
    canonical_manifest.setdefault("codebase_context", {})
    canonical_manifest.setdefault("search_intent", {})
    canonical_manifest["problem_domain"]["error_type"] = normalized_error_type
    canonical_manifest["problem_domain"]["symptom"] = symptom
    canonical_manifest["codebase_context"]["primary_slice"] = primary_slice
    canonical_manifest["codebase_context"]["architecture_pattern"] = normalized_architecture_pattern
    canonical_manifest["codebase_context"]["dependencies"] = dependencies
    canonical_manifest["search_intent"]["repomaster_query"] = repomaster_query
    canonical_manifest["search_intent"]["queries"] = queries
    canonical_manifest["search_intent"].setdefault("technical_constraints", {})
    canonical_manifest["search_intent"]["technical_constraints"]["runtime"] = normalized_runtime
    canonical_manifest["search_intent"]["technical_constraints"]["language"] = normalized_language
    canonical_manifest["search_intent"]["technical_constraints"]["required_libraries"] = required_libraries

    compatibility_signals = {
        "error_type": normalized_error_type,
        "architecture_pattern": normalized_architecture_pattern,
        "runtime": normalized_runtime,
        "language": normalized_language,
        "failing_node": failing_node,
        "symptom": symptom,
        "dependencies": dependencies,
        "required_libraries": required_libraries,
        "semantic_intent_tokens": semantic_intent_tokens,
        "repomaster_query": repomaster_query,
    }

    query_pack: dict[str, Any] = {
        "seed_queries": [],
        "template_queries": [],
        "constraint_injected_queries": [],
        "retrieval_queries": [repomaster_query],
    }
    scheduler_context_brief: dict[str, Any] = {}
    try:
        query_pack = _build_query_pack(
            repomaster_query=repomaster_query,
            seed_queries=queries,
            symptom=symptom,
            error_type=normalized_error_type,
            failing_node=failing_node,
            architecture_pattern=normalized_architecture_pattern,
            runtime=normalized_runtime,
            language=normalized_language,
            required_libraries=required_libraries,
        )
    except Exception as exc:
        warnings.append(f"query_pack generation failed; using fallback repomaster_query only: {exc}")

    compatibility_signals["query_pack"] = {
        "retrieval_queries": query_pack.get("retrieval_queries", []),
        "seed_queries": query_pack.get("seed_queries", []),
    }

    try:
        scheduler_context_brief = _build_scheduler_context_brief(
            canonical_manifest=canonical_manifest,
            compatibility_signals=compatibility_signals,
            query_pack=query_pack,
        )
    except Exception as exc:
        warnings.append(f"scheduler context brief generation failed; using fallback minimal brief: {exc}")
        scheduler_context_brief = {
            "search_intent": {
                "repomaster_query": repomaster_query,
            },
            "problem_domain": {
                "error_type": normalized_error_type,
                "symptom": symptom,
                "failing_node": failing_node,
            },
            "constraints": {
                "runtime": normalized_runtime,
                "language": normalized_language,
                "required_libraries": required_libraries,
            },
        }

    return Stage1IntakeResult(
        canonical_manifest=canonical_manifest,
        inferred_fields=inferred_fields,
        warnings=warnings,
        compatibility_signals=compatibility_signals,
        query_pack=query_pack,
        scheduler_context_brief=scheduler_context_brief,
    )


def load_stage1_acm_intake(acm_path: Path | str) -> Stage1IntakeResult:
    path = Path(acm_path)
    if not path.exists():
        raise FileNotFoundError(f"ACM input file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ACMValidationError(f"ACM JSON parse failed at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except Exception as exc:
        raise ACMValidationError(f"Failed to read ACM file: {path}. Error: {exc}") from exc

    if not isinstance(payload, dict):
        raise ACMValidationError("ACM payload must be a JSON object")

    return normalize_acm_manifest(payload)