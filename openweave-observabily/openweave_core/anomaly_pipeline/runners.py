"""
runners.py — Thin adapters that wrap the three existing detectors.

Each ``run_*`` function:
    1. Reads from the shared ``TraceBundle``.
    2. Invokes the existing engine *unchanged*.
    3. Maps its native output to a list of ``NormalizedFlag``.
    4. Returns a ``DetectorResult`` carrying flags + raw summary.

No engine is rewritten. The mapping rules are kept here so the engines stay
ignorant of pipeline-level conventions.
"""

from __future__ import annotations

from typing import Any, Optional

from openweave_core.adaptive_baseline import (
    AxisAggregator,
    MahalanobisDetector,
    MetricExtractor,
    Normalizer,
)
from openweave_core.cycle_detection import (
    build_dag_and_siblings,
    confirm_cycles,
    detect_cycles,
    sort_call_stack,
)
from openweave_core.sentinel_agent import (
    AgentNode,
    AgentSpec,
    BehaviorAnalyzer,
    Severity,
    SystemSpec,
    ToolNode,
    ToolSpec,
    build_dynamic_graph,
    build_static_graph,
    builtin_library,
)
from openweave_core.sentinel_agent.findings import FindingCategory

from openweave_core.anomaly_pipeline.contracts import (
    DetectorResult,
    NormalizedFlag,
    SOURCE_ADAPTIVE_BASELINE,
    SOURCE_CYCLE_DETECTION,
    SOURCE_SENTINEL_AGENT,
    SUBJECT_ATTACK_PATH,
    SUBJECT_EDGE,
    SUBJECT_NODE,
    SUBJECT_SPAN,
    SUBJECT_SPAN_PAIR,
    SUBJECT_TRACE,
    TraceBundle,
)


# ===========================================================================
# Adaptive baseline (AMDM) adapter
# ===========================================================================

def run_adaptive_baseline(bundle: TraceBundle) -> DetectorResult:
    """Run Steps 1-4 of the AMDM pipeline across the trace; map flags.

    Note: AMDM is a streaming detector with long warm-up. On a single trace
    in isolation it will rarely fire — flags appear once enough history has
    been processed (see WARMUP_STEPS_DEFAULT / DEFAULT_WARMUP_STEPS). The
    adapter still runs the full chain and surfaces whatever fires.
    """
    extractor = MetricExtractor()
    normalizer = Normalizer()
    aggregator = AxisAggregator()
    joint = MahalanobisDetector()

    flags: list[NormalizedFlag] = []
    n_axis_flags = 0
    n_joint_flags = 0

    for span in bundle.spans:
        raw = extractor.extract(span)
        z = normalizer.normalize(raw)
        ev = aggregator.evaluate(z)
        jr = joint.evaluate(ev)

        # Axis anomalies — one flag per triggered axis.
        for axis_name in ev.triggered_axes:
            axis = ev.axes.get(axis_name)
            if axis is None:
                continue
            flags.append(NormalizedFlag(
                id=f"amdm::axis::{span.id}::{axis_name}",
                source_pipeline=SOURCE_ADAPTIVE_BASELINE,
                category=f"axis_anomaly::{axis_name}",
                severity=Severity.WARNING,
                subject_id=span.id,
                subject_type=SUBJECT_SPAN,
                confidence=0.7,
                message=(
                    f"Axis '{axis_name}' deviated from its EWMA baseline "
                    f"(score={axis.score:+.2f}, theta={axis.baseline:+.2f}, "
                    f"|S-theta|={axis.deviation:.2f} > k*sigma={axis.threshold:.2f})."
                ),
                evidence={
                    "axis": axis_name,
                    "score": axis.score,
                    "baseline": axis.baseline,
                    "std": axis.std,
                    "deviation": axis.deviation,
                    "threshold": axis.threshold,
                    "contributing_metrics": axis.contributing_metrics,
                },
                timestamp=span.timestamp,
            ))
            n_axis_flags += 1

        # Joint anomaly — one flag per span when triggered.
        if jr.joint_anomaly:
            flags.append(NormalizedFlag(
                id=f"amdm::joint::{span.id}",
                source_pipeline=SOURCE_ADAPTIVE_BASELINE,
                category="joint_anomaly",
                severity=Severity.RISK,
                subject_id=span.id,
                subject_type=SUBJECT_SPAN,
                confidence=min(1.0, 0.6 + 0.4 * min(jr.severity / 5.0, 1.0)),
                message=(
                    f"Joint 5-axis Mahalanobis anomaly: D2={jr.d2:.2f} > "
                    f"threshold={jr.threshold:.2f} (chi2_5(0.99))."
                ),
                evidence={
                    "d2": jr.d2,
                    "threshold": jr.threshold,
                    "severity_ratio": jr.severity,
                    "state_vector": jr.state_vector,
                    "mean_vector": jr.mean_vector,
                    "n_observations": jr.n_observations,
                    "shrinkage": jr.shrinkage,
                },
                timestamp=span.timestamp,
            ))
            n_joint_flags += 1

    extractor.finalize_trace(bundle.trace_id)

    return DetectorResult(
        detector=SOURCE_ADAPTIVE_BASELINE,
        ok=True,
        flags=flags,
        raw_summary={
            "n_spans_processed": len(bundle.spans),
            "n_axis_flags": n_axis_flags,
            "n_joint_flags": n_joint_flags,
            "joint_threshold": joint.threshold,
        },
    )


# ===========================================================================
# Sentinel agent adapter
# ===========================================================================

def run_sentinel_agent(
    bundle: TraceBundle,
    *,
    system_spec: Optional[SystemSpec] = None,
) -> DetectorResult:
    """Build the dynamic graph, analyse it, map findings to flags.

    If no ``system_spec`` is provided we derive a *permissive* default from
    the observed dynamic graph (every observed agent is allowed to call every
    observed tool). This disables authorisation-based rule findings while
    keeping LLM-judge and attack-path detection fully active — appropriate
    when the analyst has not yet curated a spec for this MAS.

    The pre-built dynamic graph is cached on ``bundle.extras['dynamic_graph']``
    for any later pipeline stage that wants it.
    """
    dynamic_graph = build_dynamic_graph(bundle.spans)
    bundle.extras["dynamic_graph"] = dynamic_graph

    spec_source = "provided" if system_spec is not None else "auto_permissive"
    if system_spec is None:
        system_spec = _derive_permissive_spec(dynamic_graph)
    static_graph = build_static_graph(system_spec)

    analyzer = BehaviorAnalyzer(
        static_graph=static_graph,
        attack_library=builtin_library(),
    )
    report = analyzer.analyze(dynamic_graph)

    flags = [_finding_to_flag(f) for f in report.findings]

    return DetectorResult(
        detector=SOURCE_SENTINEL_AGENT,
        ok=True,
        flags=flags,
        raw_summary={
            "n_findings": len(report.findings),
            "overall_severity": report.overall_severity.value,
            "origin_nodes": list(report.origin_nodes),
            "propagation_paths": [list(p) for p in report.propagation_paths],
            "summary": report.summary,
            "spec_source": spec_source,
            "graph_summary": dynamic_graph.summary(),
        },
    )


def _finding_to_flag(finding) -> NormalizedFlag:
    """Map a sentinel ``Finding`` to a ``NormalizedFlag``."""
    subject_type = {
        FindingCategory.NODE: SUBJECT_NODE,
        FindingCategory.EDGE: SUBJECT_EDGE,
        FindingCategory.ATTACK_PATH: SUBJECT_ATTACK_PATH,
    }.get(finding.category, SUBJECT_TRACE)

    return NormalizedFlag(
        id=f"sentinel::{finding.id}",
        source_pipeline=SOURCE_SENTINEL_AGENT,
        category=finding.code,
        severity=finding.severity,
        subject_id=finding.subject_id,
        subject_type=subject_type,
        confidence=float(finding.confidence),
        message=finding.message,
        evidence={
            "code": finding.code,
            "source": finding.source.value,
            "remediation": finding.remediation,
            **finding.evidence,
        },
        metadata=dict(finding.metadata or {}),
    )


def _derive_permissive_spec(dynamic_graph) -> SystemSpec:
    """Build a SystemSpec where every observed agent can call every observed tool."""
    agent_ids: list[str] = []
    tool_ids: list[str] = []
    for node in dynamic_graph.nodes.values():
        if isinstance(node, AgentNode):
            agent_ids.append(node.id)
        elif isinstance(node, ToolNode):
            tool_ids.append(node.id)

    agents = [
        AgentSpec(
            id=aid,
            role="auto",
            allowed_tools=list(tool_ids),
            allowed_peers=[other for other in agent_ids if other != aid],
        )
        for aid in agent_ids
    ]
    tools = [
        ToolSpec(id=tid, name=tid, authorized_callers=list(agent_ids))
        for tid in tool_ids
    ]
    return SystemSpec(agents=agents, tools=tools, name="auto_permissive")


# ===========================================================================
# Cycle detection adapter
# ===========================================================================

def run_cycle_detection(bundle: TraceBundle) -> DetectorResult:
    """Run sort → CDCS → DAG → semantic confirmation on the parsed spans.

    Semantic confirmation calls the NVIDIA embedding API. If that fails
    (no key, network issue, quota) we degrade to *structural* flags from the
    CDCS candidates and stamp ``warning`` on the result so the report can
    surface the partial result honestly.
    """
    ct = sort_call_stack(bundle.spans)
    candidates = detect_cycles(ct)

    if not candidates:
        return DetectorResult(
            detector=SOURCE_CYCLE_DETECTION,
            ok=True,
            flags=[],
            raw_summary={
                "n_candidates": 0,
                "n_confirmed_pairs": 0,
                "label": 0,
                "stage_reached": "cdcs",
            },
        )

    # Dedupe flagged spans by id: candidates overlap heavily, so flattening
    # every first_occurrence can produce millions of duplicate references on
    # large traces. They collapse to the same sibling groups anyway.
    seen_ids: set[str] = set()
    flagged_spans = []
    for c in candidates:
        for s in c.first_occurrence:
            sid = getattr(s, "id", None) or (s.get("id") if isinstance(s, dict) else None)
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            flagged_spans.append(s)
    _, sibling_groups = build_dag_and_siblings(bundle.spans, flagged_spans)

    # Try semantic confirmation; degrade gracefully if the embedding API is unavailable.
    warning: Optional[str] = None
    confirmed_pairs: list = []
    label = 0
    try:
        result = confirm_cycles(sibling_groups)
        confirmed_pairs = result.confirmed_pairs
        label = result.label
    except (EnvironmentError, ImportError, RuntimeError) as exc:
        warning = (
            f"semantic confirmation unavailable ({type(exc).__name__}: {exc}); "
            "falling back to structural CDCS candidates."
        )

    flags: list[NormalizedFlag] = []

    if confirmed_pairs:
        for i, pair in enumerate(confirmed_pairs):
            sid_a = getattr(pair.span_a, "id", "")
            sid_b = getattr(pair.span_b, "id", "")
            flags.append(NormalizedFlag(
                id=f"cycle::confirmed::{sid_a}::{sid_b}::{i}",
                source_pipeline=SOURCE_CYCLE_DETECTION,
                category="redundant_cycle_confirmed",
                severity=Severity.RISK,
                subject_id=f"{sid_a}::{sid_b}",
                subject_type=SUBJECT_SPAN_PAIR,
                confidence=float(pair.similarity),
                message=(
                    f"Confirmed redundant cycle: cosine similarity "
                    f"{pair.similarity:.4f} between sibling spans."
                ),
                evidence={
                    "span_a_id": sid_a,
                    "span_b_id": sid_b,
                    "parent_id": pair.parent_id,
                    "similarity": pair.similarity,
                    "stage": "semantic_confirmed",
                },
            ))
    else:
        # Either confirmation came back empty OR the embedding API was
        # unavailable. Surface the structural candidates so the report still
        # explains why CDCS thought something was off.
        for i, cand in enumerate(candidates[:10]):  # cap to top-10
            first_span_id = (
                cand.first_occurrence[0].id if cand.first_occurrence else ""
            )
            sev = Severity.WARNING if warning else Severity.INFO
            flags.append(NormalizedFlag(
                id=f"cycle::candidate::{i}",
                source_pipeline=SOURCE_CYCLE_DETECTION,
                category="redundant_cycle_candidate",
                severity=sev,
                subject_id=first_span_id or f"candidate_{i}",
                subject_type=SUBJECT_SPAN if first_span_id else SUBJECT_TRACE,
                confidence=0.5,
                message=(
                    f"CDCS candidate: pattern {cand.signature} occurs "
                    f"{cand.frequency} times in this trace."
                ),
                evidence={
                    "signature": list(cand.signature),
                    "frequency": cand.frequency,
                    "length": cand.length,
                    "first_index": cand.first_index,
                    "stage": ("structural_only" if warning else "structural"),
                },
            ))

    return DetectorResult(
        detector=SOURCE_CYCLE_DETECTION,
        ok=True,
        flags=flags,
        warning=warning,
        raw_summary={
            "n_candidates": len(candidates),
            "n_confirmed_pairs": len(confirmed_pairs),
            "label": label,
            "stage_reached": (
                "semantic_confirmed" if confirmed_pairs
                else "structural_only" if warning
                else "no_structural_match"
            ),
        },
    )
