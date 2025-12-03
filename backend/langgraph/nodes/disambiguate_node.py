"""
DisambiguateNode stops execution until a user resolves ambiguous intents.
"""

from __future__ import annotations

from agents import DisambiguatorAgent
from ..state import MuseState


class DisambiguateNode:
    def __init__(self, agent: DisambiguatorAgent | None = None) -> None:
        self.agent = agent or DisambiguatorAgent()

    def __call__(self, state: MuseState) -> MuseState:
        if not state.ambiguities or state.chosen_disambiguation:
            return state

        payload = self.agent(state.fashion_query or {}, state.ambiguities)
        state.disambiguation_cards = payload.get("options", [])
        state.ui_event = payload
        state.log_event("disambiguate_node", {"cards": payload})
        return state
