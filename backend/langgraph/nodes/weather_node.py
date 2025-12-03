"""
WeatherNode fetches weather guidance using GPT-5-nano (no web search).
"""

from __future__ import annotations

from agents import WeatherAgent
from ..state import MuseState


class WeatherNode:
    def __init__(self, agent: WeatherAgent | None = None) -> None:
        self.agent = agent or WeatherAgent()

    def __call__(self, state: MuseState) -> MuseState:
        if not state.planner_plan:
            return state
        query_text = state.planner_plan.get("weather_search_query")
        if not query_text:
            return state

        result = self.agent(query_text, destination=(state.fashion_query or {}).get("destination"))
        state.weather_context = result
        state.log_event("weather_node", {"weather": result})
        return state
