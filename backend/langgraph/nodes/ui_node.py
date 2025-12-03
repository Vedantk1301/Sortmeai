"""
UINode emits frontend events such as refinement chips or disambiguation payloads.
"""

from __future__ import annotations

from agents import UIAgent
from ..state import SortmeState


class UINode:
    def __init__(self, agent: UIAgent | None = None) -> None:
        self.agent = agent or UIAgent()

    def __call__(self, state: SortmeState) -> SortmeState:
        # Preserve pending disambiguation cards until the user responds.
        if state.ui_event and state.ui_event.get("type") == "disambiguation" and not state.chosen_disambiguation:
            return state

        if state.clarification_options and not state.clarification_choice:
            state.ui_event = self.agent.clarification_cards(
                state.clarification_question or "Which one did you mean?",
                state.clarification_options,
            )
            state.log_event("ui_node", {"ui_event": state.ui_event})
            return state

        if state.mode == "capabilities_overview":
            state.ui_event = self.agent.capability_chips()
            state.log_event("ui_node", {"ui_event": state.ui_event})
            return state

        if state.outfits:
            state.ui_event = self.agent.outfit_refinements(state.outfits, state.weather_context)
            state.log_event("ui_node", {"ui_event": state.ui_event})
            return state

        if state.final_products:
            state.ui_event = self.agent.refinement_cards(state.final_products, state.fashion_query or {})
            state.log_event("ui_node", {"ui_event": state.ui_event})
        return state

