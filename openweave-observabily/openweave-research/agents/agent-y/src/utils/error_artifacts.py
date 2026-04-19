import hashlib
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def text_fingerprint(text: str, preview_chars: int = 240) -> dict[str, Any]:
    """Return stable metadata for a text payload without storing full content by default."""
    normalized = text if isinstance(text, str) else str(text)
    return {
        "length": len(normalized),
        "sha256": hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest(),
        "preview": normalized[:preview_chars],
    }


def _to_json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[truncated-depth]"

    if isinstance(value, str):
        if len(value) > 4000:
            return value[:4000] + " ...[truncated]"
        return value

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            if index >= 128:
                result["__truncated_keys__"] = len(items) - 128
                break
            result[str(key)] = _to_json_safe(item, depth + 1)
        return result

    if isinstance(value, (list, tuple, set)):
        sequence = list(value)
        result = []
        for index, item in enumerate(sequence):
            if index >= 128:
                result.append(f"[truncated_items:{len(sequence) - 128}]")
                break
            result.append(_to_json_safe(item, depth + 1))
        return result

    return repr(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_root() -> Path:
    configured = os.getenv("REPOMASTER_ERROR_ARTIFACT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _repo_root() / "error_artifacts"


def _sanitize_name(value: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value)
    return cleaned.strip("_") or "unknown"


def _write_payload(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def write_error_artifact(
    component: str,
    operation: str,
    error: Exception,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """Persist a structured JSON error artifact and return the written file path."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = uuid.uuid4().hex[:10]

    payload = {
        "artifact_version": "1.0",
        "created_at_utc": now.isoformat(),
        "component": component,
        "operation": operation,
        "runtime": {
            "cwd": os.getcwd(),
            "pid": os.getpid(),
            "python": os.sys.version,
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "args": _to_json_safe(list(error.args)),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        },
        "context": _to_json_safe(context or {}),
    }

    component_name = _sanitize_name(component)
    operation_name = _sanitize_name(operation)
    file_name = f"{timestamp}_{operation_name}_{run_id}.json"

    preferred_path = _artifact_root() / component_name / file_name
    try:
        return _write_payload(preferred_path, payload)
    except Exception as write_error:
        fallback_payload = dict(payload)
        fallback_payload["artifact_write_error"] = {
            "type": type(write_error).__name__,
            "message": str(write_error),
        }
        fallback_path = Path.cwd() / "error_artifacts_fallback" / component_name / file_name
        try:
            return _write_payload(fallback_path, fallback_payload)
        except Exception:
            return ""