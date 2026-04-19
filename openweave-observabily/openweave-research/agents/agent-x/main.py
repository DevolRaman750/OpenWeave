from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from agents.brain import Brain, BrainAnalysis, BrainError
from agents.packager import Packager, PackagerError
from config import ConfigurationError, get_greptile_settings
from fetchers.github_porter import GitHubPorter, GitHubPorterError
from fetchers.greptile_client import GreptileClient, GreptileClientError
from fetchers.tree_sitter_util import Surgeon
from schemas.acm_manifest import CompatibilityManifest
from schemas.input_packet import TraceInput

logger = logging.getLogger(__name__)


class ProcessTraceError(RuntimeError):
    """Raised when the Agent-X orchestration pipeline fails."""


async def process_trace(payload: dict[str, Any]) -> str:
    """Validate input, fetch code, reason about it, and return ACM JSON-LD."""

    raw_code = ""
    code_slice = ""
    extra_context: dict[str, Any] = {}

    try:
        trace_input = TraceInput.model_validate(payload)
        logger.info(
            "Processing trace %s for %s at %s:%s",
            trace_input.trace_id,
            trace_input.repository,
            trace_input.filepath,
            trace_input.lineno,
        )

        porter = GitHubPorter()
        raw_code = await porter.fetch_file(
            repo=trace_input.repository,
            commit_id=trace_input.commit_id,
            filepath=trace_input.filepath,
            pat=trace_input.github_pat,
        )
        logger.info(
            "Fetched source file for trace %s from %s",
            trace_input.trace_id,
            trace_input.filepath,
        )

        code_slice = Surgeon.slice_function(raw_code, trace_input.lineno)
        logger.info(
            "Extracted function slice for trace %s from line %s",
            trace_input.trace_id,
            trace_input.lineno,
        )

        brain = Brain()
        brain_analysis = await asyncio.to_thread(
            brain.evaluate_slice,
            trace_input.error_context,
            code_slice,
        )
        logger.info(
            "Brain evaluation for trace %s determined cross-file lookup=%s",
            trace_input.trace_id,
            brain_analysis.should_query_greptile,
        )

        if brain_analysis.should_query_greptile:
            greptile_settings = get_greptile_settings()
            detective = GreptileClient(
                api_key=greptile_settings.api_key,
                github_token=trace_input.github_pat,
            )
            extra_context = await detective.fetch_semantic_context(
                repo=trace_input.repository,
                commit_id=trace_input.commit_id,
                function_name=trace_input.function,
            )
            logger.info(
                "Fetched semantic detective context for trace %s",
                trace_input.trace_id,
            )
        else:
            logger.info(
                "Trace %s was explainable from the primary slice alone",
                trace_input.trace_id,
            )

        packager = Packager()
        acm = await asyncio.to_thread(
            packager.generate_acm,
            trace_input,
            code_slice,
            extra_context,
            brain_analysis,
        )
        logger.info("Generated ACM manifest for trace %s", trace_input.trace_id)
        return _serialize_manifest(acm)
    except (
        ConfigurationError,
        GitHubPorterError,
        GreptileClientError,
        BrainError,
        PackagerError,
        ValueError,
    ) as exc:
        logger.exception("Agent-X pipeline failed while processing the incoming trace.")
        raise ProcessTraceError("Failed to process trace payload.") from exc
    finally:
        raw_code = ""
        code_slice = ""
        extra_context = {}


def _serialize_manifest(acm: CompatibilityManifest) -> str:
    """Serialize with aliases so JSON-LD keeps @context and @type intact."""

    return acm.model_dump_json(by_alias=True)


def _read_payload(payload_path: Path) -> dict[str, Any]:
    """Load and validate a JSON payload file for CLI execution."""

    try:
        with payload_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Payload file was not found: {payload_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Payload file is not valid JSON: {payload_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Payload JSON root must be an object.")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Agent-X against a trace payload JSON file.",
    )
    parser.add_argument(
        "--payload",
        required=True,
        help="Path to trace payload JSON (for example: trace_2.json).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print ACM JSON output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logs while running the pipeline.",
    )
    return parser


def _hydrate_github_pat(payload: dict[str, Any]) -> dict[str, Any]:
    """Use environment PAT when payload contains a mock or missing PAT value."""

    hydrated = dict(payload)
    current_pat = hydrated.get("github_pat")
    env_pat = os.getenv("GITHUB_PAT")

    is_placeholder = not isinstance(current_pat, str) or not current_pat.strip()
    if isinstance(current_pat, str):
        normalized = current_pat.strip().lower()
        is_placeholder = is_placeholder or normalized.startswith("ghp_mock")

    if is_placeholder and env_pat:
        hydrated["github_pat"] = env_pat

    return hydrated


def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s - %(message)s",
    )

    try:
        payload = _read_payload(Path(args.payload))
        payload = _hydrate_github_pat(payload)
        acm_json = asyncio.run(process_trace(payload))
        if args.pretty:
            print(json.dumps(json.loads(acm_json), indent=2))
        else:
            print(acm_json)
        return 0
    except (ProcessTraceError, ValueError) as exc:
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
