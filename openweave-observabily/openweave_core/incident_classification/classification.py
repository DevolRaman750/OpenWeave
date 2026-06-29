"""
classification.py — Multi-label categorisation of correlated Incidents.

The classifier reads structured signals already present on each EnrichedFlag
(category code, evidence dict, resolved span/node/edge type, derived
``is_tool`` / ``is_generation`` / ``is_retrieval``) plus the incident's
aggregated payload text, and assigns one or more EvaluationCategory tags.

Design points
-------------
* **Rule-based default**: deterministic, fast, no API calls — every category
  hit records the rule name in ``Incident.category_evidence`` so the audit
  trail is explicit.
* **Pluggable LLM classifier**: ``CategoryClassifier`` is a protocol; swap
  the default for any callable that returns ``set[EvaluationCategory]``
  per incident.
* **Order**: categories are emitted in ``EvaluationCategory`` declaration
  order for deterministic output.

Rule families
-------------
PROMPT          : prompt-injection / jailbreak markers in flag.category or
                  evidence flags; payload text matches injection regexes.
TOOL_INVOCATION : flag concerns a tool node / tool edge / tool span;
                  unauthorised-tool / contract-violation codes; hallucinated
                  tool args.
LLM_GENERATION  : reasoning-edge / GENERATION-span flags; hallucination /
                  output-leak evidence; adaptive-baseline axis flags on
                  reasoning spans.
RAG             : retrieval-tool participation (is_retrieval) + any of
                  redundant cycle / unauth / tool flags routed to that span.
OBSERVABILITY   : adaptive_baseline source on any span (axis or joint).
SAFETY          : explicit safety-themed flags — secret leak, contract
                  violation, edge_judge with safety flags.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Protocol, runtime_checkable

from openweave_core.anomaly_pipeline.contracts import (
    SOURCE_ADAPTIVE_BASELINE,
    SOURCE_CYCLE_DETECTION,
    SOURCE_SENTINEL_AGENT,
)

from openweave_core.incident_classification.contracts import (
    EnrichedFlag,
    EvaluationCategory,
    Incident,
)


# ---------------------------------------------------------------------------
# Patterns / lookup tables
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_TEXT = re.compile(
    r"(?i)("
    r"ignore (?:all|any|previous|prior|the above) instructions|"
    r"disregard (?:all|any|previous|prior)|"
    r"you are now (?:dan|jailbroken|unrestricted)|"
    r"system\s*(?:prompt|message)\s*[:=]|"
    r"reveal (?:the )?system prompt"
    r")"
)

_LEAKAGE_TEXT = re.compile(
    r"(?i)("
    r"\b(?:api[_\- ]?key|secret|token|password|bearer)\s*[:=]|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"AKIA[0-9A-Z]{16}"
    r")"
)

# Category-code substrings (case-insensitive) → which evaluation bucket they
# should fire. One code can map to multiple buckets — this is multi-label.
_CODE_RULES: tuple[tuple[str, EvaluationCategory, str], ...] = (
    # (code-substring,                     category,                         rule-name)
    ("JAILBREAK",                          EvaluationCategory.PROMPT,         "code::jailbreak"),
    ("PROMPT_INJECTION",                   EvaluationCategory.PROMPT,         "code::prompt_injection"),
    ("UNAUTHORIZED_TOOL",                  EvaluationCategory.TOOL_INVOCATION,"code::unauthorized_tool"),
    ("UNAUTHORISED_TOOL",                  EvaluationCategory.TOOL_INVOCATION,"code::unauthorised_tool"),
    ("TOOL_OUTPUT_CONTRACT",               EvaluationCategory.TOOL_INVOCATION,"code::tool_contract"),
    ("HALLUCINATED_TOOL_ARGS",             EvaluationCategory.TOOL_INVOCATION,"code::tool_args_halluc"),
    ("UNSAFE_TOOL_PAYLOAD",                EvaluationCategory.TOOL_INVOCATION,"code::tool_payload_unsafe"),
    ("NODE_JUDGE_TOOL",                    EvaluationCategory.TOOL_INVOCATION,"code::tool_node_judge"),
    ("ATTACK_PATH_REDUNDANT_TOOL_CYCLE",   EvaluationCategory.TOOL_INVOCATION,"code::redundant_tool_cycle"),
    ("ATTACK_PATH_UNAUTHORISED_TOOL_USE",  EvaluationCategory.TOOL_INVOCATION,"code::unauth_pattern"),
    ("ATTACK_PATH_PROMPT_INJECTION_CHAIN", EvaluationCategory.PROMPT,         "code::injection_chain"),
    ("REDUNDANT_CYCLE",                    EvaluationCategory.TOOL_INVOCATION,"code::redundant_cycle"),
    ("AXIS_ANOMALY",                       EvaluationCategory.OBSERVABILITY,  "code::axis_anomaly"),
    ("JOINT_ANOMALY",                      EvaluationCategory.OBSERVABILITY,  "code::joint_anomaly"),
)

# Evidence-flag substrings (in JudgeVerdict / Finding evidence dicts) →
# which bucket should fire.
_EVIDENCE_FLAG_RULES: tuple[tuple[str, EvaluationCategory, str], ...] = (
    ("jailbreak",               EvaluationCategory.PROMPT,         "evidence::jailbreak"),
    ("prompt_injection",        EvaluationCategory.PROMPT,         "evidence::prompt_injection"),
    ("hallucinated_tool_args",  EvaluationCategory.TOOL_INVOCATION,"evidence::tool_args_halluc"),
    ("unsafe_tool_payload",     EvaluationCategory.TOOL_INVOCATION,"evidence::tool_payload_unsafe"),
    ("unauthorised_relation",   EvaluationCategory.TOOL_INVOCATION,"evidence::unauth_relation"),
    ("potential_secret_leak",   EvaluationCategory.SAFETY,         "evidence::secret_leak"),
    ("tool_output_leak",        EvaluationCategory.SAFETY,         "evidence::tool_leak"),
    ("payload_leak",            EvaluationCategory.SAFETY,         "evidence::payload_leak"),
)


# ---------------------------------------------------------------------------
# Classifier protocol + default implementation
# ---------------------------------------------------------------------------

@runtime_checkable
class CategoryClassifier(Protocol):
    """Anything that can label an Incident with EvaluationCategories.

    Implementations must return a tuple ``(categories, evidence)`` where
    ``categories`` is the multi-label set and ``evidence[c]`` is a list of
    rule / model identifiers that fired for category ``c``.
    """

    def classify(
        self, incident: Incident
    ) -> tuple[set[EvaluationCategory], dict[EvaluationCategory, list[str]]]:
        ...


class RuleBasedClassifier:
    """Default rule-based multi-label classifier.

    Reads:
      * flag.category (substring matches in ``_CODE_RULES``)
      * flag.evidence['flags'] (substring matches in ``_EVIDENCE_FLAG_RULES``)
      * EnrichedFlag.is_tool / is_generation / is_retrieval
      * source_pipeline (adaptive_baseline → OBSERVABILITY by default)
      * payload_texts (injection / leakage regexes)
    """

    def classify(
        self, incident: Incident
    ) -> tuple[set[EvaluationCategory], dict[EvaluationCategory, list[str]]]:
        cats: set[EvaluationCategory] = set()
        evidence: dict[EvaluationCategory, list[str]] = {}

        def _hit(cat: EvaluationCategory, rule: str) -> None:
            cats.add(cat)
            evidence.setdefault(cat, []).append(rule)

        for ef in incident.flags:
            self._apply_code_rules(ef, _hit)
            self._apply_evidence_flag_rules(ef, _hit)
            self._apply_subject_shape_rules(ef, _hit)
            self._apply_source_pipeline_rules(ef, _hit)
            self._apply_text_rules(ef, _hit)

        return cats, evidence

    # ------------------------------------------------------------------
    # Rule families
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_code_rules(ef: EnrichedFlag, hit) -> None:
        code = (ef.flag.category or "").upper()
        for token, category, rule_name in _CODE_RULES:
            if token in code:
                hit(category, rule_name)

    @staticmethod
    def _apply_evidence_flag_rules(ef: EnrichedFlag, hit) -> None:
        flags_in_evidence = ef.flag.evidence.get("flags") or []
        if not isinstance(flags_in_evidence, list):
            return
        normalised = [str(x).lower() for x in flags_in_evidence]
        for needle, category, rule_name in _EVIDENCE_FLAG_RULES:
            if any(needle in n for n in normalised):
                hit(category, rule_name)

    @staticmethod
    def _apply_subject_shape_rules(ef: EnrichedFlag, hit) -> None:
        # Retrieval semantics win first — incidents centred on retrieval get
        # routed to RAG even if other categories also apply.
        if ef.is_retrieval:
            hit(EvaluationCategory.RAG, "shape::retrieval_tool")
        if ef.is_tool:
            hit(EvaluationCategory.TOOL_INVOCATION, "shape::tool")
        if ef.is_generation:
            hit(EvaluationCategory.LLM_GENERATION, "shape::generation")

    @staticmethod
    def _apply_source_pipeline_rules(ef: EnrichedFlag, hit) -> None:
        if ef.flag.source_pipeline == SOURCE_ADAPTIVE_BASELINE:
            hit(EvaluationCategory.OBSERVABILITY, "source::adaptive_baseline")
        if ef.flag.source_pipeline == SOURCE_CYCLE_DETECTION:
            # Cycle flags imply something tool-shaped repeats.
            hit(EvaluationCategory.TOOL_INVOCATION, "source::cycle_detection")
        # Sentinel can produce any of the buckets; rules above handle that.
        _ = SOURCE_SENTINEL_AGENT  # documentation reference

    @staticmethod
    def _apply_text_rules(ef: EnrichedFlag, hit) -> None:
        for text in ef.payload_texts:
            if not text:
                continue
            if _PROMPT_INJECTION_TEXT.search(text):
                hit(EvaluationCategory.PROMPT, "text::injection_marker")
                hit(EvaluationCategory.SAFETY, "text::injection_marker")
            if _LEAKAGE_TEXT.search(text):
                hit(EvaluationCategory.SAFETY, "text::secret_marker")


# ---------------------------------------------------------------------------
# Applier — attaches classifier output to incidents in place
# ---------------------------------------------------------------------------

def classify_incidents(
    incidents: Iterable[Incident],
    *,
    classifier: Optional[CategoryClassifier] = None,
) -> list[Incident]:
    """Run the classifier over each incident; attach categories + evidence.

    Returns the same list (mutated in place) for fluent chaining.
    """
    classifier = classifier or RuleBasedClassifier()
    out: list[Incident] = []
    for inc in incidents:
        cats, evidence = classifier.classify(inc)
        # Stable order: declaration order of EvaluationCategory enum.
        ordered = [c for c in EvaluationCategory if c in cats]
        inc.categories = ordered
        inc.category_evidence = evidence
        out.append(inc)
    return out
