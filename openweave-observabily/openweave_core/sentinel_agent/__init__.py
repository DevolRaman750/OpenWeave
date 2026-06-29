"""
sentinel_agent — Graph-based, LLM-powered anomaly detection for AI agent systems.

Phase 1: Dual interaction-graph construction
    Static Graph  (cold-start)  : intended topology from system specification,
                                  including authorization boundaries.
    Dynamic Graph (runtime)     : actual interactions reconstructed from live
                                  ParsedSpan traces by the EventMonitor.

Phase 2: Behavior Analyzer (core evaluation engine)
    Module 1 — Local anomalies via decoupled node + edge evaluators using a
               dual layer (deterministic rules + LLM-as-a-judge).
    Module 2 — Systemic failures via Weisfeiler-Lehman × Edge-Distance
               kernels matched against an AttackPathLibrary, with semantic
               correlation and a stricter double-check fallback.
    Risk     — RiskReport with root-cause attribution + tailored remediations.

Top-level usage
---------------
    from openweave_core.sentinel_agent import (
        SystemSpec, AgentSpec, ToolSpec,
        build_static_graph, build_dynamic_graph,
        BehaviorAnalyzer, builtin_library,
    )

    static  = build_static_graph(spec)
    dynamic = build_dynamic_graph(spans)
    report  = BehaviorAnalyzer(
        static_graph=static,
        attack_library=builtin_library(),
    ).analyze(dynamic)
"""

# Phase 1 — schema / graph
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    AgentSpec,
    Edge,
    EdgeType,
    Node,
    NodeType,
    SystemSpec,
    ToolNode,
    ToolSpec,
)
from openweave_core.sentinel_agent.graph import InteractionGraph, merge
from openweave_core.sentinel_agent.static_graph import build_static_graph
from openweave_core.sentinel_agent.dynamic_graph import (
    EventMonitor,
    SpanResolver,
    build_dynamic_graph,
)

# Phase 2 — findings / judges / evaluators / matcher / analyzer
from openweave_core.sentinel_agent.findings import (
    Finding,
    FindingCategory,
    FindingSource,
    JudgeVerdict,
    Remediation,
    RiskReport,
    Severity,
)
from openweave_core.sentinel_agent.llm_judge import (
    EdgeContext,
    EdgeJudge,
    HeuristicEdgeJudge,
    HeuristicNodeJudge,
    NodeContext,
    NodeJudge,
    NullEdgeJudge,
    NullNodeJudge,
)
from openweave_core.sentinel_agent.rule_engine import RuleBasedEvaluator
from openweave_core.sentinel_agent.status_evaluator import (
    DEFAULT_RISK_THRESHOLD,
    STRICT_RISK_THRESHOLD,
    EdgeEvaluator,
    NodeEvaluator,
)
from openweave_core.sentinel_agent.wl_kernel import (
    DEFAULT_BLEND,
    DEFAULT_DEPTH,
    DEFAULT_TIME_DECAY_SECONDS,
    StructuralScore,
    edge_distance_kernel,
    normalised_edge_distance_kernel,
    normalised_wl_kernel,
    structural_similarity,
    wl_relabel,
    wl_subtree_kernel,
)
from openweave_core.sentinel_agent.attack_paths import (
    AttackPath,
    AttackPathLibrary,
    SemanticSignature,
    builtin_library,
)
from openweave_core.sentinel_agent.path_matcher import (
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_STRUCTURAL_THRESHOLD,
    AttackPathMatcher,
    PathMatch,
)
from openweave_core.sentinel_agent.behavior_analyzer import BehaviorAnalyzer

__all__ = [
    # Phase 1 — schema
    "AgentNode", "ToolNode", "Node", "NodeType", "Edge", "EdgeType",
    "AgentSpec", "ToolSpec", "SystemSpec",
    # Phase 1 — graphs
    "InteractionGraph", "merge",
    "build_static_graph",
    "EventMonitor", "SpanResolver", "build_dynamic_graph",
    # Phase 2 — output types
    "Finding", "FindingCategory", "FindingSource",
    "JudgeVerdict", "Remediation", "RiskReport", "Severity",
    # Phase 2 — judges
    "NodeJudge", "EdgeJudge",
    "NodeContext", "EdgeContext",
    "HeuristicNodeJudge", "HeuristicEdgeJudge",
    "NullNodeJudge", "NullEdgeJudge",
    # Phase 2 — rule + status evaluators
    "RuleBasedEvaluator",
    "NodeEvaluator", "EdgeEvaluator",
    "DEFAULT_RISK_THRESHOLD", "STRICT_RISK_THRESHOLD",
    # Phase 2 — kernels
    "wl_relabel", "wl_subtree_kernel", "normalised_wl_kernel",
    "edge_distance_kernel", "normalised_edge_distance_kernel",
    "structural_similarity", "StructuralScore",
    "DEFAULT_DEPTH", "DEFAULT_TIME_DECAY_SECONDS", "DEFAULT_BLEND",
    # Phase 2 — attack paths
    "AttackPath", "AttackPathLibrary", "SemanticSignature", "builtin_library",
    # Phase 2 — matcher + analyzer
    "AttackPathMatcher", "PathMatch",
    "DEFAULT_STRUCTURAL_THRESHOLD", "DEFAULT_SEMANTIC_THRESHOLD",
    "BehaviorAnalyzer",
]
