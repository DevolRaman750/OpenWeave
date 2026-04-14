from __future__ import annotations

import asyncio
import logging
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
