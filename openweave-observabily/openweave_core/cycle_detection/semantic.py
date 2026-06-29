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
from openweave_core.resilience import CircuitBreaker

SpanLike = Union[ParsedSpan, dict[str, Any]]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL = "nvidia/nv-embedcode-7b-v1"
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
SIMILARITY_THRESHOLD: float = 0.83  # phi — tune here if needed
_ENV_KEY = "NVIDIA_API_KEY"

# --- external-call resilience (bound the NVIDIA endpoint) ------------------
# Per-call hang-up + bounded retries with backoff (the SDK applies exponential
# backoff between retries automatically). A degraded endpoint now fails in
# ~timeout instead of the SDK's multi-minute default.
EMBED_TIMEOUT_SECONDS: float = float(os.environ.get("OPENWEAVE_EMBED_TIMEOUT", "10"))
EMBED_MAX_RETRIES: int = int(os.environ.get("OPENWEAVE_EMBED_MAX_RETRIES", "3"))

# Process-wide circuit breaker guarding the shared embedding endpoint: after
# this many consecutive failures, calls fail fast (CircuitOpenError, a
# RuntimeError) for the cooldown window instead of each trace paying the
# timeout. confirm_cycles()'s existing degrade path catches RuntimeError, so a
# tripped breaker transparently falls back to exact-match / structural flags.
_EMBED_BREAKER = CircuitBreaker(
    name="nvidia-embed",
    failure_threshold=int(os.environ.get("OPENWEAVE_EMBED_BREAKER_THRESHOLD", "5")),
    recovery_timeout=float(os.environ.get("OPENWEAVE_EMBED_BREAKER_COOLDOWN", "30")),
)


def embed_breaker_state() -> str:
    """Current embedding circuit-breaker state (for health endpoints/diagnostics)."""
    return _EMBED_BREAKER.state.value

# Scaling guards (see scripts/scale_sweep.py). A redundant loop only needs a
# *sample* of a sibling group to be confirmed, so we cap the per-group work to
# keep semantic confirmation O(cap^2) regardless of how wide the group is.
DEFAULT_MAX_GROUP_SIZE: int = 64   # max distinct sibling texts compared per group
EMBED_BATCH_SIZE: int = 96         # texts per embedding request (chunked)
# Global budget on distinct texts embedded per trace. Exact-match (byte-identical
# redundancy) runs uncapped on every group first, so this only bounds the
# *paraphrase* near-duplicate check; genuine loops are pervasive and surface
# within the budget. Keeps a benign large trace to ~1-2 embedding round-trips.
GLOBAL_EMBED_BUDGET: int = 192


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

        # timeout = per-call hang-up; max_retries = bounded retries w/ backoff.
        self._client = OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=EMBED_TIMEOUT_SECONDS,
            max_retries=EMBED_MAX_RETRIES,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, chunked to a safe per-request size.

        Empty strings are filtered before the request and their slots are
        returned as empty lists so the caller's index alignment is preserved.
        The non-empty texts are split into ``EMBED_BATCH_SIZE`` chunks so a
        single logical ``embed()`` never overruns the endpoint's input cap;
        each chunk is one HTTP request and results are stitched back by index.

        Parameters
        ----------
        texts : raw output strings, one per sibling span.

        Returns
        -------
        list[list[float]]
            Parallel to *texts*; empty list for spans with no output text.

        Raises
        ------
        RuntimeError      On any API-level failure (network, quota, bad model).
        CircuitOpenError  (a RuntimeError) when the breaker is tripped — the
                          endpoint is failing, so the call is rejected instantly.
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
        valid_texts = list(valid_texts)

        # Route the whole (chunked) network operation through the breaker so a
        # single logical embed() counts as one success/failure.
        embeddings = _EMBED_BREAKER.call(self._embed_valid, valid_texts)
        for pos, vec in enumerate(embeddings):
            vectors[orig_idxs[pos]] = vec
        return vectors

    def _embed_valid(self, valid_texts: list[str]) -> list[list[float]]:
        """Embed already-non-empty texts in EMBED_BATCH_SIZE chunks.

        Returns a list parallel to *valid_texts*. Raises RuntimeError on any
        API failure (after the client's bounded retries are exhausted).
        """
        out: list[list[float]] = [[] for _ in valid_texts]
        for off in range(0, len(valid_texts), EMBED_BATCH_SIZE):
            chunk = valid_texts[off:off + EMBED_BATCH_SIZE]
            try:
                response = self._client.embeddings.create(
                    input=chunk,
                    model=EMBED_MODEL,
                    encoding_format="float",
                    extra_body={"input_type": "passage", "truncate": "END"},
                )
            except Exception as exc:
                raise RuntimeError(
                    f"nv-embedcode-7b-v1 API call failed: {exc}"
                ) from exc

            for item in response.data:
                out[off + item.index] = item.embedding

        return out


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


def _cosine_matrix(vectors: list[list[float]]):
    """Full pairwise cosine-similarity matrix via one normalized matmul.

    Returns a ``(k, k)`` numpy array, or ``None`` if numpy is unavailable (the
    caller then falls back to scalar ``_cosine``). Each row is L2-normalized,
    so ``S = M_norm @ M_norm.T`` yields cosine similarities in one BLAS call —
    replacing the O(k^2) Python pair loop that caused the scaling cliff.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover — numpy is a hard dep in practice
        return None

    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # avoid divide-by-zero; zero vectors stay zero
    normalized = matrix / norms
    return normalized @ normalized.T


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
# Confirmation helpers
# ---------------------------------------------------------------------------

def _exact_pairs(
    siblings: list[SpanLike], parent_id: str
) -> list[ConfirmedPair]:
    """Confirm byte-identical sibling outputs without any embedding call.

    Redundant loops emit identical outputs, so any output text seen twice is a
    confirmed redundant pair at similarity 1.0. O(k) hashing over the full
    group — emits one representative pair (first two occurrences) per duplicate
    text bucket. Works even when the embedding endpoint is unreachable.
    """
    first_seen: dict[str, SpanLike] = {}
    emitted: set[str] = set()
    pairs: list[ConfirmedPair] = []
    for span in siblings:
        text = _output_text(span)
        if not text.strip():
            continue
        if text in first_seen:
            if text not in emitted:
                pairs.append(ConfirmedPair(
                    span_a=first_seen[text], span_b=span,
                    similarity=1.0, parent_id=parent_id,
                ))
                emitted.add(text)
        else:
            first_seen[text] = span
    return pairs


def _distinct_reps(siblings: list[SpanLike]) -> list[tuple[str, SpanLike]]:
    """Ordered (text, representative-span) for each distinct non-empty output."""
    reps: list[tuple[str, SpanLike]] = []
    seen: set[str] = set()
    for span in siblings:
        text = _output_text(span)
        if not text.strip() or text in seen:
            continue
        seen.add(text)
        reps.append((text, span))
    return reps


def _dedup_sort(pairs: list[ConfirmedPair]) -> list[ConfirmedPair]:
    """Collapse duplicate span-pairs (keep highest similarity), sort stably."""
    best: dict[tuple[str, str], ConfirmedPair] = {}
    for pair in pairs:
        key = tuple(sorted((_span_id(pair.span_a), _span_id(pair.span_b))))
        current = best.get(key)
        if current is None or pair.similarity > current.similarity:
            best[key] = pair
    result = list(best.values())
    result.sort(key=lambda p: (
        -p.similarity, _span_id(p.span_a), _span_id(p.span_b),
    ))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def confirm_cycles(
    sibling_groups: list[SiblingGroup],
    *,
    embed_client: Optional[EmbedClient] = None,
    threshold: float = SIMILARITY_THRESHOLD,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
) -> SemanticResult:
    """Run semantic confirmation on CDCS-flagged sibling groups (scalable).

    Three-pass design that keeps cost bounded regardless of trace size:

    1. **Exact match** (per group, full): identical sibling outputs are
       confirmed in O(k) via hashing — no embedding, no O(k^2).
    2. **Single batched embedding**: every remaining *distinct* output text
       across *all* groups is embedded in one chunked call (was one call per
       group), de-duplicated by text.
    3. **Vectorized cosine** (per group, capped to ``max_group_size`` distinct
       texts): one normalized matmul yields the similarity matrix; pairs above
       ``threshold`` are confirmed.

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
    max_group_size:
        Cap on distinct sibling texts compared per group (bounds O(cap^2)).

    Returns
    -------
    SemanticResult
        ``label=1`` and non-empty ``confirmed_pairs`` if any pair exceeds phi
        (or exact duplicates exist). ``label=0`` with empty list otherwise.

    Raises
    ------
    TypeError    If *sibling_groups* is not a list.
    RuntimeError Propagated from EmbedClient on API failure *only* when there
                 are non-identical candidates to embed and no exact-duplicate
                 pairs were found (so the runner still degrades structurally).
    """
    if not isinstance(sibling_groups, list):
        raise TypeError(
            f"sibling_groups must be list, got {type(sibling_groups).__name__}"
        )

    if not sibling_groups:
        return SemanticResult(label=0)

    exact_pairs: list[ConfirmedPair] = []
    # Per-group distinct representatives that still need semantic comparison.
    group_reps: list[tuple[str, list[tuple[str, SpanLike]]]] = []
    distinct_texts: dict[str, None] = {}  # insertion-ordered set across groups

    for group in sibling_groups:
        # Stable order so caps and pair selection are deterministic.
        siblings = sorted(group.siblings, key=_span_id)
        if len(siblings) < 2:
            continue

        group_exact = _exact_pairs(siblings, group.parent_id)
        exact_pairs.extend(group_exact)

        # Byte-identical outputs are already strong redundancy evidence, so a
        # group confirmed by exact match skips the embedding round-trip — this
        # makes the common redundant-loop case zero-embedding (sub-second).
        # Only all-distinct groups embed, to catch paraphrase-style redundancy.
        if group_exact:
            continue

        reps = _distinct_reps(siblings)
        if len(reps) > max_group_size:
            reps = reps[:max_group_size]
        if len(reps) >= 2:
            group_reps.append((group.parent_id, reps))
            for text, _span in reps:
                distinct_texts.setdefault(text, None)

    # No cross-text comparisons needed -> exact-match result is the answer.
    if not group_reps:
        confirmed = _dedup_sort(exact_pairs)
        return SemanticResult(label=1 if confirmed else 0, confirmed_pairs=confirmed)

    # Embed every distinct text once, bounded by the global budget. Degrade to
    # exact-only if the API fails. (Texts beyond the budget simply aren't
    # embedded; their groups fall back to exact-match coverage.)
    text_list = list(distinct_texts.keys())[:GLOBAL_EMBED_BUDGET]
    try:
        client = embed_client or EmbedClient()
        vectors = client.embed(text_list)
    except (EnvironmentError, ImportError, RuntimeError):
        if exact_pairs:
            confirmed = _dedup_sort(exact_pairs)
            return SemanticResult(label=1, confirmed_pairs=confirmed)
        raise  # nothing salvageable -> let run_cycle_detection fall back
    text_to_vec = dict(zip(text_list, vectors))

    semantic_pairs: list[ConfirmedPair] = []
    for parent_id, reps in group_reps:
        usable = [(t, s) for (t, s) in reps if text_to_vec.get(t)]
        if len(usable) < 2:
            continue
        sims = _cosine_matrix([text_to_vec[t] for t, _ in usable])
        if sims is None:  # numpy missing -> scalar fallback (bounded by cap)
            for (i, (ti, si)), (j, (tj, sj)) in combinations(enumerate(usable), 2):
                sim = _cosine(text_to_vec[ti], text_to_vec[tj])
                if sim > threshold:
                    semantic_pairs.append(ConfirmedPair(si, sj, round(sim, 6), parent_id))
            continue
        import numpy as np
        iu, ju = np.triu_indices(len(usable), k=1)
        for idx in np.nonzero(sims[iu, ju] > threshold)[0]:
            i, j = int(iu[idx]), int(ju[idx])
            semantic_pairs.append(ConfirmedPair(
                span_a=usable[i][1], span_b=usable[j][1],
                similarity=round(float(sims[i, j]), 6), parent_id=parent_id,
            ))

    confirmed = _dedup_sort(exact_pairs + semantic_pairs)
    return SemanticResult(label=1 if confirmed else 0, confirmed_pairs=confirmed)
