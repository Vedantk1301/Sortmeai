"""
StylistNode produces the final narrative response using validated products only.
"""

from __future__ import annotations

from agents import StylistAgent
from ..state import SortmeState


class StylistNode:
    def __init__(self, agent: StylistAgent | None = None) -> None:
        self.agent = agent or StylistAgent()

    def __call__(self, state: SortmeState) -> SortmeState:
        clarification = {
            "options": state.clarification_options,
            "choice": state.clarification_choice,
            "question": state.clarification_question,
        }

        state.stylist_response = self.agent(
            state.final_products or [],
            state.fashion_query or {},
            outfits=state.outfits,
            weather=state.weather_context,
            trends=state.trends_context,
            clarification=clarification,
            mode=state.mode,
        )
        mode = state.mode or ("outfits" if state.outfits else ("products" if state.final_products else "clarification"))
        state.log_event("stylist_node", {"response_preview": (state.stylist_response or "")[:120], "mode": mode})
        return state

