"""
ClarifierNode triggers clarification cards for ambiguous queries.
"""

from __future__ import annotations

from agents import ClarifierAgent
from ..state import SortmeState


class ClarifierNode:
    def __init__(self, agent: ClarifierAgent | None = None) -> None:
        self.agent = agent or ClarifierAgent()

    def __call__(self, state: SortmeState) -> SortmeState:
        if not state.fashion_query or state.fashion_query.get("query_type") != "specific":
            return state

        raw_query = state.fashion_query.get("raw_query") or state.user_message

        # If already clarified for this query, skip
        if state.clarification_choice and state.clarification_source_query == raw_query:
            return state

        result = self.agent(raw_query, state.fashion_query)
        if not result.get("needs_clarification"):
            return state

        state.clarification_options = result.get("options", [])
        state.clarification_question = result.get("question") or "Which one did you mean?"
        state.clarification_source_query = raw_query
        state.ui_event = {
            "type": "clarification_cards",
            "question": state.clarification_question,
            "options": state.clarification_options,
        }
        state.log_event(
            "clarifier_node",
            {"question": state.clarification_question, "options": len(state.clarification_options)},
        )
        return state

