"""
semantic.py — Semantic Similarity Confirmation (final stage, Hybrid Approach).

Pipeline position
-----------------
DAG sibling groups
  -> confirm_cycles(sibling_groups)
  -> SemanticResult { label: f(T) in {0,1}, confirmed_pairs }
  -> observability dashboard

Algorithm
---------
  Step 1 : Iterate sibling groups only  (not full trajectory)
  Step 2 : Embed span output_text via nv-embedcode-7b-v1  -> vi in R^d
  Step 3 : Cosine similarity cos(vi, vj) for every sibling pair
  Step 4 : cos(vi, vj) > phi=0.83  ->  f(T)=1  (bad cycle confirmed)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from dataclasses import dataclass, field

# Load .env from this package directory. override=True so the embedding key
# here wins over any NVIDIA_API_KEY injected by the parser bootstrap.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass  # dotenv optional — key can be set directly in the shell environment

from itertools import combinations
from typing import Any, Optional, Union

from openweave_core.cycle_detection.dag import SiblingGroup
from openweave_core.models.span import ParsedSpan

SpanLike = Union[ParsedSpan, dict[str, Any]]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL = "nvidia/nv-embedcode-7b-v1"
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
SIMILARITY_THRESHOLD: float = 0.83  # phi — tune here if needed
_ENV_KEY = "NVIDIA_API_KEY"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class ConfirmedPair:
    """A sibling pair whose semantic similarity exceeds phi.

    Attributes
    ----------
    span_a, span_b : the two redundant spans
    similarity     : cos(va, vb) — value that triggered confirmation
    parent_id      : their shared parent node in Gt
    """

    span_a: SpanLike
    span_b: SpanLike
    similarity: float
    parent_id: str


@dataclass
class SemanticResult:
    """Final classification output of the Hybrid Approach.

    Attributes
    ----------
    label            : f(T) — 0 = clean trace, 1 = bad cycle confirmed
    confirmed_pairs  : pairs that exceeded phi (empty when label=0)
    """

    label: int  # f(T) in {0, 1}
    confirmed_pairs: list[ConfirmedPair] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Embedding client — NVIDIA NIM (OpenAI-compatible interface)
# ---------------------------------------------------------------------------

class EmbedClient:
    """Wrapper around the NVIDIA NIM embeddings endpoint for nv-embedcode-7b-v1.

    Parameters
    ----------
    api_key  : NVIDIA API key. Falls back to ``NVIDIA_API_KEY`` env var.
    base_url : NIM inference base URL (override for self-hosted NIM).

    Raises
    ------
    ImportError        If ``openai`` package is not installed.
    EnvironmentError   If no API key is found.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _NVIDIA_BASE_URL,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required: pip install openai"
            ) from exc

        key = api_key or os.environ.get(_ENV_KEY)
        if not key:
            raise EnvironmentError(
                f"NVIDIA API key not found. Set the {_ENV_KEY} environment variable "
                "or pass api_key= to EmbedClient."
            )

        self._client = OpenAI(api_key=key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single API call.

        Empty strings are filtered before the request and their slots are
        returned as empty lists so the caller's index alignment is preserved.

        Parameters
        ----------
        texts : raw output strings, one per sibling span.

        Returns
        -------
        list[list[float]]
            Parallel to *texts*; empty list for spans with no output text.

        Raises
        ------
        RuntimeError  On any API-level failure (network, quota, bad model).
        """
        if not texts:
            return []

        non_empty: list[tuple[int, str]] = [
            (i, t) for i, t in enumerate(texts) if t.strip()
        ]

        vectors: list[list[float]] = [[] for _ in texts]

        if not non_empty:
            return vectors

        orig_idxs, valid_texts = zip(*non_empty)

        try:
            response = self._client.embeddings.create(
                input=list(valid_texts),
                model=EMBED_MODEL,
                encoding_format="float",
                extra_body={"input_type": "passage", "truncate": "END"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"nv-embedcode-7b-v1 API call failed: {exc}"
            ) from exc

        for item in response.data:
            vectors[orig_idxs[item.index]] = item.embedding

        return vectors


# ---------------------------------------------------------------------------
# Step 3 — Cosine similarity
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors.

    Uses numpy when available for speed; falls back to pure math.
    Returns 0.0 for zero-norm or empty vectors.
    """
    if not a or not b:
        return 0.0

    try:
        import numpy as np

        va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom > 0.0 else 0.0

    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        denom = norm_a * norm_b
        return dot / denom if denom > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Span field helpers
# ---------------------------------------------------------------------------

def _output_text(span: SpanLike) -> str:
    if isinstance(span, ParsedSpan):
        return span.output_text or ""
    return span.get("output_text") or span.get("output") or ""


def _span_id(span: SpanLike) -> str:
    if isinstance(span, ParsedSpan):
        return span.id
    return span.get("id") or span.get("span_id") or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def confirm_cycles(
    sibling_groups: list[SiblingGroup],
    *,
    embed_client: Optional[EmbedClient] = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> SemanticResult:
    """Run semantic confirmation on CDCS-flagged sibling groups.

    For each SiblingGroup, all sibling spans are embedded in a single batched
    API call. Every unique pair (si, sj) within the group is evaluated against
    phi. All confirmed pairs are returned for dashboard surfacing.

    Parameters
    ----------
    sibling_groups:
        Output of ``find_sibling_groups`` / ``build_dag_and_siblings``.
        Groups with fewer than 2 siblings are silently skipped.
    embed_client:
        Pre-constructed EmbedClient. Created from NVIDIA_API_KEY env var
        if not provided.
    threshold:
        phi — cosine similarity cutoff (default 0.83, strict inequality).

    Returns
    -------
    SemanticResult
        ``label=1`` and non-empty ``confirmed_pairs`` if any pair exceeds phi.
        ``label=0`` with empty list otherwise.

    Raises
    ------
    TypeError    If *sibling_groups* is not a list.
    RuntimeError Propagated from EmbedClient on API failure.
    """
    if not isinstance(sibling_groups, list):
        raise TypeError(
            f"sibling_groups must be list, got {type(sibling_groups).__name__}"
        )

    if not sibling_groups:
        return SemanticResult(label=0)

    client = embed_client or EmbedClient()
    confirmed: list[ConfirmedPair] = []

    for group in sibling_groups:
        siblings = group.siblings
        if len(siblings) < 2:
            continue

        texts = [_output_text(s) for s in siblings]
        vectors = client.embed(texts)

        for (i, si), (j, sj) in combinations(enumerate(siblings), 2):
            vi, vj = vectors[i], vectors[j]
            if not vi or not vj:
                continue

            sim = _cosine(vi, vj)

            if sim > threshold:  # phi = 0.83, strict inequality
                confirmed.append(
                    ConfirmedPair(
                        span_a=si,
                        span_b=sj,
                        similarity=round(sim, 6),
                        parent_id=group.parent_id,
                    )
                )

    confirmed.sort(key=lambda p: p.similarity, reverse=True)

    return SemanticResult(
        label=1 if confirmed else 0,
        confirmed_pairs=confirmed,
    )
