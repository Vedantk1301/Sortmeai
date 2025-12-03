"""
Node for broad-intent planning: generates product, web, and weather queries.
"""

from __future__ import annotations

from agents import KnowledgePlannerAgent
from ..state import MuseState


class KnowledgePlannerNode:
    def __init__(self, agent: KnowledgePlannerAgent | None = None) -> None:
        self.agent = agent or KnowledgePlannerAgent()

    def __call__(self, state: MuseState) -> MuseState:
        if not state.fashion_query or state.fashion_query.get("query_type") != "broad":
            return state

        plan = self.agent(state.fashion_query)
        state.planner_plan = plan
        state.broad_intent = state.fashion_query
        state.log_event("knowledge_planner_node", {"plan": plan})
        return state
