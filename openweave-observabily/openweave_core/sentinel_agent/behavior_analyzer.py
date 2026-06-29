"""
behavior_analyzer.py — Phase-2 top-level orchestrator.

Wires together:

    Module 1 (local)     : NodeEvaluator + EdgeEvaluator (rule + judge layers)
    Module 2 (systemic)  : AttackPathMatcher (WL × Edge-Distance kernels)
    Double-check         : if Module 2 fired but Module 1 didn't, re-run
                           judges on the matched subgraph with a stricter
                           threshold.
    Risk Responder       : assemble RiskReport with root-cause attribution +
                           tailored remediation list.

Usage
-----
    analyzer = BehaviorAnalyzer(
        static_graph=static_graph,
        attack_library=builtin_library(),
        node_judge=my_granite_guardian_wrapper,    # optional
        edge_judge=my_llamafirewall_wrapper,        # optional
    )

    report = analyzer.analyze(dynamic_graph)
    if report.has_critical():
        page_oncall(report)
    elif report.has_anomaly():
        ticket(report)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from openweave_core.sentinel_agent.attack_paths import AttackPathLibrary
from openweave_core.sentinel_agent.findings import (
    Finding,
    FindingCategory,
    FindingSource,
    Remediation,
    RiskReport,
    Severity,
)
from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.llm_judge import EdgeJudge, NodeJudge
from openweave_core.sentinel_agent.path_matcher import (
    AttackPathMatcher,
    PathMatch,
)
from openweave_core.sentinel_agent.rule_engine import RuleBasedEvaluator
from openweave_core.sentinel_agent.schema import Edge, EdgeType
from openweave_core.sentinel_agent.status_evaluator import (
    DEFAULT_RISK_THRESHOLD,
    EdgeEvaluator,
    NodeEvaluator,
    STRICT_RISK_THRESHOLD,
)


class BehaviorAnalyzer:
    """End-to-end Phase-2 evaluator.

    Parameters
    ----------
    static_graph     : InteractionGraph from Phase 1 (kind="static").
    attack_library   : library of known risky topologies / inefficiency loops.
    node_judge       : optional LLM-as-a-judge for nodes. Defaults to heuristic.
    edge_judge       : optional LLM-as-a-judge for edges. Defaults to heuristic.
    risk_threshold   : verdict score ≥ this triggers a Finding in normal mode.
    strict_threshold : lower threshold used during the double-check pass.
    matcher          : pre-built AttackPathMatcher; constructed internally if None.
    """

    def __init__(
        self,
        *,
        static_graph: InteractionGraph,
        attack_library: AttackPathLibrary,
        node_judge: Optional[NodeJudge] = None,
        edge_judge: Optional[EdgeJudge] = None,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
        strict_threshold: float = STRICT_RISK_THRESHOLD,
        matcher: Optional[AttackPathMatcher] = None,
    ) -> None:
        self._rules = RuleBasedEvaluator(static_graph)
        self._node_eval = NodeEvaluator(
            rule_engine=self._rules,
            node_judge=node_judge,
            risk_threshold=risk_threshold,
        )
        self._edge_eval = EdgeEvaluator(
            rule_engine=self._rules,
            edge_judge=edge_judge,
            risk_threshold=risk_threshold,
        )
        self._matcher = matcher or AttackPathMatcher(library=attack_library)
        self._strict_threshold = strict_threshold

    # -----------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------

    def analyze(self, dynamic_graph: InteractionGraph) -> RiskReport:
        if dynamic_graph.kind != "dynamic":
            raise ValueError(
                f"dynamic_graph.kind must be 'dynamic', got {dynamic_graph.kind!r}"
            )

        # Annotate unauthorised edges up-front so the path matcher's semantic
        # signature can see them without an extra static-graph lookup.
        self._annotate_unauthorised_edges(dynamic_graph)

        # -- Module 1 ----------------------------------------------------
        node_findings = self._node_eval.evaluate_all(dynamic_graph)
        edge_findings = self._edge_eval.evaluate_all(dynamic_graph)

        local_findings = node_findings + edge_findings

        # -- Module 2 ----------------------------------------------------
        path_matches: list[PathMatch] = self._matcher.search(
            dynamic_graph, require_match=True
        )
        path_findings = [m.finding for m in path_matches if m.finding is not None]

        # -- Double-check ------------------------------------------------
        # If the matcher fired but Module 1 produced no RISK/CRITICAL local
        # findings, re-run judges on the matched subgraph with a stricter
        # threshold. This catches covert anomalies that hide below the normal
        # risk gate.
        local_high = [
            f for f in local_findings
            if f.severity in (Severity.RISK, Severity.CRITICAL)
        ]
        double_check_findings: list[Finding] = []
        if path_matches and not local_high:
            double_check_findings = self._double_check(
                dynamic_graph, path_matches
            )

        all_findings = (
            local_findings + path_findings + double_check_findings
        )

        # -- Risk Responder ---------------------------------------------
        return self._build_report(dynamic_graph, all_findings, path_matches)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _annotate_unauthorised_edges(self, graph: InteractionGraph) -> None:
        """Tag dynamic edges that violate static auth so semantic correlator can see them."""
        for edge in graph.edges.values():
            if edge.edge_type is EdgeType.TOOL_INVOCATION:
                allowed = self._rules.static_graph.is_tool_call_allowed(
                    edge.source, edge.target
                )
                edge.metadata["rule_unauthorised"] = not allowed
            elif edge.edge_type is EdgeType.MESSAGE:
                allowed = self._rules.static_graph.is_message_allowed(
                    edge.source, edge.target
                )
                edge.metadata["rule_unauthorised"] = not allowed

    def _double_check(
        self,
        graph: InteractionGraph,
        path_matches: list[PathMatch],
    ) -> list[Finding]:
        """Re-run judges on the matched subgraph with the strict threshold."""
        # Union of all nodes/edges referenced by matched paths
        node_ids: set[str] = set()
        edge_ids: set[str] = set()
        for m in path_matches:
            node_ids.update(m.origin_node_ids)
            edge_ids.update(m.matched_edge_ids)

        # Fall back to whole graph if no subjects were identified
        if not node_ids and not edge_ids:
            node_ids = set(graph.nodes.keys())
            edge_ids = set(graph.edges.keys())

        out: list[Finding] = []
        for nid in node_ids:
            node = graph.get_node(nid)
            if node is None:
                continue
            out.extend(self._node_eval.evaluate(
                node, graph, threshold_override=self._strict_threshold,
            ))
        if edge_ids:
            out.extend(self._edge_eval.evaluate_all(
                graph,
                threshold_override=self._strict_threshold,
                only_edge_ids=edge_ids,
            ))

        # Mark each finding's source as double-check (the evaluators already
        # do this when threshold_override is set, but tag remediations clearly).
        return out

    # -----------------------------------------------------------------
    # Risk Responder — assemble the final report
    # -----------------------------------------------------------------

    def _build_report(
        self,
        graph: InteractionGraph,
        findings: list[Finding],
        path_matches: list[PathMatch],
    ) -> RiskReport:
        # Dedupe by id (a finding might appear twice if both layers fired)
        unique: dict[str, Finding] = {}
        for f in findings:
            if f.id not in unique:
                unique[f.id] = f
        deduped = list(unique.values())

        overall = self._overall_severity(deduped)

        # Root-cause attribution
        origin_nodes = self._infer_origin_nodes(deduped, path_matches, graph)
        propagation = self._infer_propagation_paths(origin_nodes, graph)

        # Tailored remediations
        remediations = self._collect_remediations(deduped)

        summary = self._summarise(deduped, overall, path_matches)
        trace_id = (
            next(iter(graph.nodes.values())).metadata.get("trace_id", "")
            if graph.nodes else ""
        )

        return RiskReport(
            trace_id=trace_id,
            findings=sorted(
                deduped,
                key=lambda f: _SEVERITY_RANK[f.severity],
                reverse=True,
            ),
            remediations=remediations,
            origin_nodes=origin_nodes,
            propagation_paths=propagation,
            summary=summary,
            overall_severity=overall,
            metadata={
                "n_findings": len(deduped),
                "n_path_matches": len(path_matches),
                "graph_summary": graph.summary(),
            },
        )

    def _overall_severity(self, findings: list[Finding]) -> Severity:
        if not findings:
            return Severity.INFO
        return Severity.max(*[f.severity for f in findings])

    def _infer_origin_nodes(
        self,
        findings: list[Finding],
        path_matches: list[PathMatch],
        graph: InteractionGraph,
    ) -> list[str]:
        """Origin = nodes implicated by the highest-severity findings.

        Priority order:
          1. Nodes referenced by attack-path matches.
          2. Sources of CRITICAL/RISK edge findings.
          3. Subjects of CRITICAL/RISK node findings.
        """
        seen: set[str] = set()
        order: list[str] = []

        def _add(node_id: str) -> None:
            if node_id and node_id in graph.nodes and node_id not in seen:
                seen.add(node_id)
                order.append(node_id)

        for m in path_matches:
            for nid in m.origin_node_ids:
                _add(nid)

        for f in findings:
            if f.severity not in (Severity.RISK, Severity.CRITICAL):
                continue
            if f.category is FindingCategory.EDGE:
                edge = graph.edges.get(f.subject_id)
                if edge:
                    _add(edge.source)
            elif f.category is FindingCategory.NODE:
                _add(f.subject_id)

        return order

    def _infer_propagation_paths(
        self,
        origin_nodes: list[str],
        graph: InteractionGraph,
    ) -> list[list[str]]:
        """For each origin, return up to one chronologically-ordered downstream chain."""
        paths: list[list[str]] = []
        for origin in origin_nodes:
            chain = self._chronological_chain(origin, graph, max_depth=5)
            if chain:
                paths.append(chain)
        return paths

    @staticmethod
    def _chronological_chain(
        start: str, graph: InteractionGraph, max_depth: int = 5
    ) -> list[str]:
        chain = [start]
        visited = {start}
        current = start
        for _ in range(max_depth):
            outs = sorted(
                graph.outgoing_edges(current),
                key=lambda e: e.timestamp,
            )
            advanced = False
            for e in outs:
                if e.target not in visited:
                    chain.append(e.target)
                    visited.add(e.target)
                    current = e.target
                    advanced = True
                    break
            if not advanced:
                break
        return chain if len(chain) > 1 else []

    def _collect_remediations(self, findings: list[Finding]) -> list[Remediation]:
        seen_keys: set[tuple[str, str]] = set()
        rems: list[Remediation] = []
        for f in findings:
            if not f.remediation:
                continue
            key = (f.subject_id, f.remediation)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rems.append(Remediation(
                target_id=f.subject_id,
                action=_action_for_code(f.code),
                description=f.remediation,
                priority=f.severity,
            ))
        # Sort by priority (CRITICAL first)
        rems.sort(key=lambda r: _SEVERITY_RANK[r.priority], reverse=True)
        return rems

    def _summarise(
        self,
        findings: list[Finding],
        overall: Severity,
        path_matches: list[PathMatch],
    ) -> str:
        if not findings:
            return "No anomalies detected."
        by_cat: dict[FindingCategory, int] = defaultdict(int)
        for f in findings:
            by_cat[f.category] += 1
        parts = [
            f"Overall severity: {overall.value.upper()}.",
            f"{len(findings)} finding(s) across "
            + ", ".join(f"{n} {c.value}" for c, n in by_cat.items())
            + ".",
        ]
        if path_matches:
            names = ", ".join(m.path.name for m in path_matches)
            parts.append(f"Matched attack patterns: {names}.")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.RISK: 2,
    Severity.CRITICAL: 3,
}


def _action_for_code(code: str) -> str:
    """Map a finding code to a short action tag for the remediation list."""
    c = code.upper()
    if "UNAUTHORIZED" in c or "UNAUTHORISED" in c:
        return "restrict_permission"
    if "JAILBREAK" in c or "INJECTION" in c:
        return "refine_prompt"
    if "LEAK" in c or "SECRET" in c:
        return "scrub_payload"
    if "CONTRACT" in c:
        return "validate_schema"
    if "ATTACK_PATH" in c:
        return "block_pattern"
    if "UNKNOWN" in c:
        return "declare_in_spec"
    return "investigate"
