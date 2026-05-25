from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from base64 import b64encode
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from openweave_core.models.span import ParsedSpan


def _bootstrap_local_environment() -> None:
    """Make local CLI usage work without manually exporting env vars.

    This keeps credentials out of source code while still letting:
    `python -m openweave_core.parser.trace_fetcher <trace_id>`
    work from the `openweave-observabily` checkout.
    """

    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    sdk_path = workspace_root / "OpenWeave-SDK"
    env_path = repo_root / "openweave-research" / "agents" / ".env"

    if sdk_path.exists():
        sdk_path_string = str(sdk_path)
        if sdk_path_string not in sys.path:
            sys.path.insert(0, sdk_path_string)

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


_bootstrap_local_environment()


_MISSING = object()


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        current = _get_one(value, name, _MISSING)
        if current is not _MISSING:
            return current
    return default


def _get_one(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict) and name in value:
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=False)
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None

    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _usage_value(observation: Any, *keys: str) -> int:
    usage_details = _as_dict(_get(observation, "usage_details", "usageDetails"))
    usage = _as_dict(_get(observation, "usage"))

    for key in keys:
        value = usage_details.get(key)
        if value is not None:
            return _as_int(value)
        value = usage.get(key)
        if value is not None:
            return _as_int(value)

    payload_keys = {"input", "output", "prompt", "completion", "total"}
    for key in keys:
        if key in payload_keys:
            continue
        value = _get(observation, key)
        if value is not None:
            return _as_int(value)

    return 0


def _cost_value(observation: Any) -> Optional[float]:
    cost_details = _as_dict(_get(observation, "cost_details", "costDetails"))
    for key in ("total", "total_cost", "totalCost"):
        if cost_details.get(key) is not None:
            return _as_float(cost_details[key])

    for name in (
        "total_cost",
        "totalCost",
        "calculated_total_cost",
        "calculatedTotalCost",
        "total_price",
        "totalPrice",
    ):
        value = _get(observation, name)
        if value is not None:
            return _as_float(value)

    input_cost = _as_float(_get(observation, "input_cost", "inputCost"))
    output_cost = _as_float(_get(observation, "output_cost", "outputCost"))
    if input_cost is not None or output_cost is not None:
        return (input_cost or 0.0) + (output_cost or 0.0)
    return None


def _latency_value(observation: Any, start_time: Any, end_time: Any) -> Optional[float]:
    latency = _as_float(_get(observation, "latency"))
    if latency is not None:
        return latency
    parsed_start = _as_datetime(start_time)
    parsed_end = _as_datetime(end_time)
    if parsed_start is not None and parsed_end is not None:
        return (parsed_end - parsed_start).total_seconds()
    return None


def _sort_key(span: Any) -> tuple[int, str]:
    timestamp = _as_datetime(span.timestamp)
    if timestamp is not None:
        return (0, timestamp.isoformat())
    return (1, span.id)


def _depth_for(span_id: str, parent_by_id: dict[str, Optional[str]]) -> int:
    depth = 0
    seen = {span_id}
    parent_id = parent_by_id.get(span_id)

    while parent_id and parent_id in parent_by_id and parent_id not in seen:
        seen.add(parent_id)
        depth += 1
        parent_id = parent_by_id.get(parent_id)

    return depth


def _fetch_trace_with_http(trace_id: str) -> dict[str, Any]:
    base_url = (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or "https://cloud.langfuse.com"
    ).rstrip("/")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required")

    url = f"{base_url}/api/public/traces/{trace_id}?fields=core,io,observations,metrics"
    auth = b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Langfuse HTTP API returned {exc.code}: {body}") from exc


def parse_langfuse_trace(trace_id: str) -> list[ParsedSpan]:
    """Fetch a Langfuse trace and return sorted, structured span objects."""

    trace = _fetch_trace_with_http(trace_id)
    observations = _get(trace, "observations", default=[]) or []

    spans = []
    for observation in observations:
        span_id = str(_get(observation, "id", default=""))
        input_text = _as_text(_get(observation, "input"))
        output_text = _as_text(_get(observation, "output"))
        start_time = _as_datetime(_get(observation, "start_time", "startTime"))
        end_time = _as_datetime(_get(observation, "end_time", "endTime"))
        input_tokens = _usage_value(
            observation, "input", "prompt", "prompt_tokens", "promptTokens", "input_tokens", "inputTokens"
        )
        output_tokens = _usage_value(
            observation,
            "output",
            "completion",
            "completion_tokens",
            "completionTokens",
            "output_tokens",
            "outputTokens",
        )
        total_tokens = _usage_value(observation, "total", "total_tokens", "totalTokens")

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        spans.append(
            ParsedSpan(
                id=span_id,
                trace_id=str(_get(observation, "trace_id", "traceId", default=trace_id) or trace_id),
                span_type=str(_get(observation, "type", default="")),
                tool_name=_get(observation, "name"),
                input_text=input_text,
                output_text=output_text,
                input_hash=hashlib.md5(input_text.encode("utf-8")).hexdigest(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency=_latency_value(observation, start_time, end_time),
                cost=_cost_value(observation),
                timestamp=start_time,
                start_time=start_time,
                end_time=end_time,
                parent_id=_get(
                    observation,
                    "parent_observation_id",
                    "parentObservationId",
                ),
                model=_get(observation, "model", "provided_model_name", "providedModelName"),
                metadata=_get(observation, "metadata"),
                status_message=_get(observation, "status_message", "statusMessage"),
                level=str(_get(observation, "level", default="") or "") or None,
            )
        )

    children_by_parent: dict[str, list[str]] = {}
    parent_by_id = {span.id: span.parent_id for span in spans}

    for span in spans:
        if span.parent_id:
            children_by_parent.setdefault(span.parent_id, []).append(span.id)

    for span in spans:
        span.child_ids = children_by_parent.get(span.id, [])
        span.depth = _depth_for(span.id, parent_by_id)

    return sorted(spans, key=_sort_key)


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m openweave_core.parser.trace_fetcher <trace_id>")

    trace_id_arg = sys.argv[1]
    host = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
    print(f"Fetching trace {trace_id_arg} from {host or 'configured Langfuse host'}...", file=sys.stderr)

    try:
        parsed_spans = parse_langfuse_trace(trace_id_arg)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Check that Langfuse is running, the trace ID exists, and the API is reachable.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(json.dumps([asdict(span) for span in parsed_spans], indent=2, default=_json_default))
