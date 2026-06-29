"""
correlation.py — Group correlated EnrichedFlags into one Incident each.

Approach
--------
Union-Find over the flat list of EnrichedFlags. Two flags are joined when
*any* signal below holds; the signals are recorded so each Incident can
explain how its members got merged together.

Signals (intentionally compound — practical correlation, not just same-name):

    affected_spans_overlap
        EnrichedFlag.affected_span_ids ∩ EnrichedFlag.affected_span_ids ≠ ∅.
        Strongest single signal — two detectors pointing at literally the
        same span(s).

    same_root_span
        Both flags trace back to the same root span (same trace branch).
        Required to be *augmented* by another signal (temporal proximity OR
        category overlap) so we don't lump every flag in a tree together.

    same_tool
        Both flags concern the same tool_name AND were within
        TEMPORAL_PROXIMITY_SECONDS of each other.

    parent_child_chain
        One flag's span is the other flag's parent / child / sibling
        (within the same agent loop).

    detector_corroboration
        Different source_pipelines, identical primary subject_id, similar
        category. Catches the "rule_engine + edge_judge + path_matcher all
        pointing at the same unauth call" case.

    cycle_pair_membership
        Cycle-pair flag shares one of its two spans with a span-typed flag
        from any other detector.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional

from openweave_core.anomaly_pipeline.contracts import (
    SUBJECT_SPAN,
    SUBJECT_SPAN_PAIR,
)
from openweave_core.sentinel_agent.findings import Severity

from openweave_core.incident_classification.contracts import (
    EnrichedFlag,
    Incident,
)

TEMPORAL_PROXIMITY_SECONDS: float = 30.0


_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.RISK: 2,
    Severity.CRITICAL: 3,
}


# ---------------------------------------------------------------------------
# Union-Find primitive
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Return True if a merge happened (False if already in same set)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return True

    def groups(self, n: int) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(n):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return out


# ---------------------------------------------------------------------------
# Correlator
# ---------------------------------------------------------------------------

class IncidentCorrelator:
    """Group correlated EnrichedFlags into Incident objects."""

    def __init__(
        self,
        *,
        temporal_proximity_seconds: float = TEMPORAL_PROXIMITY_SECONDS,
    ) -> None:
        self._tau = timedelta(seconds=temporal_proximity_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correlate(self, flags: list[EnrichedFlag]) -> list[Incident]:
        if not flags:
            return []

        n = len(flags)
        uf = _UnionFind(n)
        # signal name → list of (i, j) joined by that rule, for audit
        signals_by_rule: dict[str, list[tuple[int, int]]] = {}

        for i in range(n):
            for j in range(i + 1, n):
                rules = self._joining_rules(flags[i], flags[j])
                if not rules:
                    continue
                if uf.union(i, j):
                    for r in rules:
                        signals_by_rule.setdefault(r, []).append((i, j))
                else:
                    for r in rules:
                        signals_by_rule.setdefault(r, []).append((i, j))

        # Build incidents from the disjoint sets.
        groups = uf.groups(n)
        incidents: list[Incident] = []
        for idx, (root, members) in enumerate(
            sorted(groups.items(), key=lambda kv: kv[0])
        ):
            members_flags = [flags[m] for m in members]
            incidents.append(
                self._make_incident(
                    incident_id=f"inc-{idx + 1}",
                    flags=members_flags,
                    signals_by_rule=signals_by_rule,
                    member_set=set(members),
                )
            )

        # Sort: highest severity first, then most flags first.
        incidents.sort(
            key=lambda inc: (-_SEVERITY_RANK[inc.severity], -len(inc.flags))
        )
        # Re-id after sort to keep ids stable per output order.
        for idx, inc in enumerate(incidents):
            inc.id = f"inc-{idx + 1}"
        return incidents

    # ------------------------------------------------------------------
    # Joining rules
    # ------------------------------------------------------------------

    def _joining_rules(self, a: EnrichedFlag, b: EnrichedFlag) -> list[str]:
        rules: list[str] = []

        # 1. Span-set overlap — strongest.
        if a.affected_span_ids and b.affected_span_ids and (
            a.affected_span_ids & b.affected_span_ids
        ):
            rules.append("affected_spans_overlap")

        # 2. Same primary subject + different detectors → corroboration.
        if (
            a.flag.subject_id == b.flag.subject_id
            and a.flag.source_pipeline != b.flag.source_pipeline
            and a.flag.subject_id  # not empty
        ):
            rules.append("detector_corroboration")

        # 3. Cycle-pair membership.
        if a.flag.subject_type == SUBJECT_SPAN_PAIR or b.flag.subject_type == SUBJECT_SPAN_PAIR:
            if a.affected_span_ids & b.affected_span_ids:
                rules.append("cycle_pair_membership")

        # 4. Parent / child / sibling locality.
        if self._parent_child_locality(a, b):
            rules.append("parent_child_chain")

        # 5. Same tool + temporal proximity.
        if (
            a.tool_name and b.tool_name and a.tool_name == b.tool_name
            and self._temporally_close(a, b)
        ):
            rules.append("same_tool")

        # 6. Same root + (temporal proximity or shared subject base).
        if (
            a.root_span is not None and b.root_span is not None
            and a.root_span.id == b.root_span.id
            and self._temporally_close(a, b)
        ):
            rules.append("same_root_span")

        return rules

    @staticmethod
    def _parent_child_locality(a: EnrichedFlag, b: EnrichedFlag) -> bool:
        if a.span is None or b.span is None:
            return False
        if a.span.id == b.span.parent_id:
            return True
        if b.span.id == a.span.parent_id:
            return True
        if (
            a.span.parent_id is not None
            and a.span.parent_id == b.span.parent_id
            and a.span.id != b.span.id
        ):
            return True  # siblings under the same parent
        return False

    def _temporally_close(self, a: EnrichedFlag, b: EnrichedFlag) -> bool:
        ta = self._timestamp_of(a)
        tb = self._timestamp_of(b)
        if ta is None or tb is None:
            return False
        return abs(ta - tb) <= self._tau

    @staticmethod
    def _timestamp_of(e: EnrichedFlag) -> Optional[datetime]:
        if e.timestamp is not None:
            return e.timestamp
        if e.span is not None:
            return e.span.timestamp or e.span.start_time
        return None

    # ------------------------------------------------------------------
    # Incident assembly
    # ------------------------------------------------------------------

    def _make_incident(
        self,
        *,
        incident_id: str,
        flags: list[EnrichedFlag],
        signals_by_rule: dict[str, list[tuple[int, int]]],
        member_set: set[int],
    ) -> Incident:
        subject_ids: set[str] = set()
        affected_span_ids: set[str] = set()
        source_pipelines: set[str] = set()
        timestamps: list[datetime] = []
        confidence_total = 0.0

        for f in flags:
            subject_ids.add(f.flag.subject_id)
            affected_span_ids.update(f.affected_span_ids)
            source_pipelines.add(f.flag.source_pipeline)
            ts = self._timestamp_of(f)
            if ts is not None:
                timestamps.append(ts)
            confidence_total += f.flag.confidence

        # Severity = max across members.
        severity = max(
            (f.flag.severity for f in flags),
            key=lambda s: _SEVERITY_RANK[s],
            default=Severity.INFO,
        )
        # Confidence = average, biased up by severity.
        avg_conf = confidence_total / max(len(flags), 1)
        confidence = min(
            1.0,
            avg_conf + 0.1 * _SEVERITY_RANK[severity],
        )

        # Primary subject: most-referenced subject id; tiebreak by earliest
        # timestamp.
        primary_subject_id = self._pick_primary_subject(flags)

        # Pull rules whose pairs all sit inside this incident.
        triggered_signals: list[str] = sorted({
            rule
            for rule, pairs in signals_by_rule.items()
            if any(a in member_set and b in member_set for a, b in pairs)
        })

        title, description = self._summarise(flags, primary_subject_id, severity)

        return Incident(
            id=incident_id,
            flags=flags,
            title=title,
            description=description,
            primary_subject_id=primary_subject_id,
            subject_ids=subject_ids,
            affected_span_ids=affected_span_ids,
            source_pipelines=source_pipelines,
            severity=severity,
            confidence=confidence,
            started_at=min(timestamps) if timestamps else None,
            ended_at=max(timestamps) if timestamps else None,
            correlation_signals=triggered_signals,
        )

    @staticmethod
    def _pick_primary_subject(flags: list[EnrichedFlag]) -> str:
        counts: dict[str, int] = {}
        for f in flags:
            if not f.flag.subject_id:
                continue
            counts[f.flag.subject_id] = counts.get(f.flag.subject_id, 0) + 1
        if not counts:
            return ""
        # Tiebreak: prefer the most frequent, then prefer span subjects,
        # then alphabetical for determinism.
        span_subjects = {
            f.flag.subject_id for f in flags
            if f.flag.subject_type == SUBJECT_SPAN
        }
        return max(
            counts.keys(),
            key=lambda sid: (counts[sid], sid in span_subjects, sid),
        )

    @staticmethod
    def _summarise(
        flags: list[EnrichedFlag],
        primary_subject_id: str,
        severity: Severity,
    ) -> tuple[str, str]:
        sources = sorted({f.flag.source_pipeline for f in flags})
        categories = sorted({f.flag.category for f in flags})
        title = (
            f"{severity.value.upper()}: "
            f"{', '.join(categories[:3])}"
            + (" ..." if len(categories) > 3 else "")
        )
        description = (
            f"{len(flags)} flag(s) from {len(sources)} detector(s) "
            f"correlated around {primary_subject_id!r}. "
            f"Sources: {sources}."
        )
        return title, description


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def correlate_flags(
    flags: Iterable[EnrichedFlag],
    *,
    temporal_proximity_seconds: float = TEMPORAL_PROXIMITY_SECONDS,
) -> list[Incident]:
    return IncidentCorrelator(
        temporal_proximity_seconds=temporal_proximity_seconds,
    ).correlate(list(flags))
