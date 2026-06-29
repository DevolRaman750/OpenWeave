"""
static_graph.py — Build the cold-start Static Graph from a SystemSpec.

The static graph captures the *intended* topology of the MAS: which agents
exist, which tools exist, and exactly which interaction paths are permitted
before any execution has occurred.

Output is an ``InteractionGraph`` populated with:
    nodes               — one per AgentSpec / ToolSpec
    edges               — one per *allowed* interaction (TOOL_INVOCATION or
                          MESSAGE), stamped with ``datetime.min`` so they sort
                          before any dynamic edge.
    allowed_tool_calls,
    allowed_messages,
    authorized_callers  — populated authorisation tables for Phase-2 checks.

The builder reconciles the two views of tool-call authorisation:
``AgentSpec.allowed_tools`` and ``ToolSpec.authorized_callers``. By default we
take the *union* (either source granting permission is enough); this can be
tightened to the intersection with ``policy="strict"`` for orgs that require
both sides to agree.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, Literal

from openweave_core.sentinel_agent.graph import InteractionGraph
from openweave_core.sentinel_agent.schema import (
    AgentNode,
    AgentSpec,
    Edge,
    EdgeType,
    SystemSpec,
    ToolNode,
    ToolSpec,
)

log = logging.getLogger(__name__)

# Stable, far-past sentinel for static-edge timestamps so chronological
# sorts always place them before dynamic-graph edges.
_STATIC_TS: datetime = datetime.min

Policy = Literal["union", "strict"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_static_graph(
    spec: SystemSpec,
    *,
    policy: Policy = "union",
    log_inconsistencies: bool = True,
) -> InteractionGraph:
    """Construct the Static Graph from a complete ``SystemSpec``.

    Parameters
    ----------
    spec:
        The configuration / specification of the MAS. Must contain at least
        one AgentSpec; tools are optional.
    policy:
        How to reconcile ``AgentSpec.allowed_tools`` with
        ``ToolSpec.authorized_callers`` for the SAME (agent, tool) pair.

        * ``"union"``  — either side granting is enough. (Default.)
        * ``"strict"`` — both sides must list the relationship; mismatches
                         become *unauthorised* edges (not added).
    log_inconsistencies:
        Emit WARNING logs when the two authorisation views disagree.

    Returns
    -------
    InteractionGraph
        Fully populated static graph: nodes, allowed-edge entries, and the
        three authorisation tables (``allowed_tool_calls``,
        ``allowed_messages``, ``authorized_callers``).

    Raises
    ------
    TypeError
        If *spec* is not a ``SystemSpec``.
    ValueError
        If duplicate ids are found within agents or within tools, or if an
        agent and a tool share an id.
    """
    if not isinstance(spec, SystemSpec):
        raise TypeError(f"spec must be SystemSpec, got {type(spec).__name__}")

    _validate_no_id_collisions(spec.agents, spec.tools)

    g = InteractionGraph(kind="static")

    _add_agent_nodes(g, spec.agents)
    _add_tool_nodes(g, spec.tools)
    _add_tool_invocation_edges(
        g, spec, policy=policy, log_inconsistencies=log_inconsistencies
    )
    _add_message_edges(g, spec)

    return g


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _validate_no_id_collisions(
    agents: Iterable[AgentSpec], tools: Iterable[ToolSpec]
) -> None:
    seen_agent_ids: set[str] = set()
    for a in agents:
        if a.id in seen_agent_ids:
            raise ValueError(f"Duplicate AgentSpec id: {a.id!r}")
        seen_agent_ids.add(a.id)

    seen_tool_ids: set[str] = set()
    for t in tools:
        if t.id in seen_tool_ids:
            raise ValueError(f"Duplicate ToolSpec id: {t.id!r}")
        seen_tool_ids.add(t.id)

    overlap = seen_agent_ids & seen_tool_ids
    if overlap:
        raise ValueError(
            f"Agent and tool ids must not overlap; offending: {sorted(overlap)}"
        )


def _add_agent_nodes(g: InteractionGraph, agents: Iterable[AgentSpec]) -> None:
    for spec in agents:
        g.add_node(
            AgentNode(
                id=spec.id,
                role=spec.role,
                system_prompt=spec.system_prompt,
                declared_capabilities=list(spec.capabilities),
                metadata=dict(spec.metadata),
            )
        )


def _add_tool_nodes(g: InteractionGraph, tools: Iterable[ToolSpec]) -> None:
    for spec in tools:
        g.add_node(
            ToolNode(
                id=spec.id,
                name=spec.name or spec.id,
                description=spec.description,
                input_constraints=dict(spec.input_constraints),
                output_constraints=dict(spec.output_constraints),
                behavioral_contract=spec.behavioral_contract,
                metadata=dict(spec.metadata),
            )
        )


def _add_tool_invocation_edges(
    g: InteractionGraph,
    spec: SystemSpec,
    *,
    policy: Policy,
    log_inconsistencies: bool,
) -> None:
    """Cross-reference agent.allowed_tools with tool.authorized_callers."""
    agent_view: dict[str, set[str]] = {
        a.id: set(a.allowed_tools) for a in spec.agents
    }
    tool_view: dict[str, set[str]] = {
        t.id: set(t.authorized_callers) for t in spec.tools
    }
    tool_ids: set[str] = set(tool_view)
    agent_ids: set[str] = set(agent_view)

    for agent_id in agent_ids:
        for tool_id in tool_view.keys() | agent_view[agent_id]:
            agent_says = tool_id in agent_view[agent_id]
            tool_says = agent_id in tool_view.get(tool_id, set())

            if not (agent_says or tool_says):
                continue  # neither side grants — skip

            if tool_id not in tool_ids:
                # Agent references a tool that wasn't declared in spec.tools
                if log_inconsistencies:
                    log.warning(
                        "Agent %s lists unknown tool %s in allowed_tools",
                        agent_id, tool_id,
                    )
                continue

            permitted: bool
            if policy == "union":
                permitted = agent_says or tool_says
            elif policy == "strict":
                permitted = agent_says and tool_says
            else:
                raise ValueError(f"Unknown policy: {policy!r}")

            if log_inconsistencies and agent_says != tool_says:
                log.warning(
                    "Authorisation mismatch for (agent=%s, tool=%s): "
                    "AgentSpec.allowed_tools=%s, ToolSpec.authorized_callers=%s",
                    agent_id, tool_id, agent_says, tool_says,
                )

            if not permitted:
                continue

            # Record both authorisation views.
            g.allowed_tool_calls[agent_id].add(tool_id)
            g.authorized_callers[tool_id].add(agent_id)

            # Materialise a static edge representing the allowed path.
            edge_id = f"static::{agent_id}->{tool_id}::tool_invocation"
            g.add_edge(
                Edge(
                    id=edge_id,
                    source=agent_id,
                    target=tool_id,
                    edge_type=EdgeType.TOOL_INVOCATION,
                    timestamp=_STATIC_TS,
                    metadata={"static": True, "policy": policy},
                )
            )


def _add_message_edges(g: InteractionGraph, spec: SystemSpec) -> None:
    """Allowed agent-to-agent message paths from ``AgentSpec.allowed_peers``."""
    agent_ids: set[str] = {a.id for a in spec.agents}

    for agent in spec.agents:
        for peer_id in agent.allowed_peers:
            if peer_id == agent.id:
                continue  # ignore self-loops at the spec level
            if peer_id not in agent_ids:
                log.warning(
                    "Agent %s lists unknown peer %s in allowed_peers",
                    agent.id, peer_id,
                )
                continue

            g.allowed_messages[agent.id].add(peer_id)

            edge_id = f"static::{agent.id}->{peer_id}::message"
            g.add_edge(
                Edge(
                    id=edge_id,
                    source=agent.id,
                    target=peer_id,
                    edge_type=EdgeType.MESSAGE,
                    timestamp=_STATIC_TS,
                    metadata={"static": True},
                )
            )
