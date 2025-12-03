"""
OutfitBuilderNode composes outfits from the pooled validated products.
"""

from __future__ import annotations

from agents import OutfitBuilderAgent
from ..state import SortmeState


class OutfitBuilderNode:
    def __init__(self, agent: OutfitBuilderAgent | None = None) -> None:
        self.agent = agent or OutfitBuilderAgent()

    def __call__(self, state: SortmeState) -> SortmeState:
        if not state.pooled_valid_products:
            return state
        context = {
            "destination": (state.fashion_query or {}).get("destination"),
            "occasion": (state.fashion_query or {}).get("occasion"),
            "gender": (state.fashion_query or {}).get("gender"),
            "weather": state.weather_context,
            "fashion_rules": (state.fashion_knowledge or {}).get("rules"),
        }
        outfits = self.agent(state.pooled_valid_products, context).get("outfits", [])
        state.outfits = outfits
        state.log_event("outfit_builder_node", {"outfits": len(outfits)})
        return state

