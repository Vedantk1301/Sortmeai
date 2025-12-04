"""
ParseNode converts raw user text into the structured fashion_query.
"""

from __future__ import annotations

import logging
from agents import ParserAgent
from config import Config
from ..state import SortmeState


class ParseNode:
    def __init__(self, parser: ParserAgent | None = None) -> None:
        self.parser = parser or ParserAgent()
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: SortmeState) -> SortmeState:
        self.logger.info(f"[PARSE] Processing user message: '{state.user_message}'")
        primary = self.parser(state.user_message)
        confidence = float(primary.get("confidence", 0.0))

        # Escalate to orchestrator model if confidence is low
        if confidence < 0.55 and Config.ORCHESTRATOR_MODEL and Config.ORCHESTRATOR_MODEL != Config.FAST_MODEL:
            try:
                secondary = self.parser(state.user_message, model_override=Config.ORCHESTRATOR_MODEL)
                secondary_conf = float(secondary.get("confidence", 0.0))
                if secondary_conf > confidence:
                    primary = secondary
                    confidence = secondary_conf
                    self.logger.info(f"[PARSE] Escalated to orchestrator model -> confidence {confidence:.2f}")
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(f"[PARSE] Orchestrator parse failed: {exc}")

        fashion_query = primary
        state.intent_confidence = confidence
        state.last_raw_query = fashion_query.get("raw_query") or state.user_message
        state.recent_intent = fashion_query.get("query_type") or fashion_query.get("intent")
        state.recent_destination = fashion_query.get("destination") or state.recent_destination
        state.recent_occasion = fashion_query.get("occasion") or state.recent_occasion
        state.recent_gender = fashion_query.get("gender") or state.recent_gender

        # Lightweight context hints for downstream nodes
        fashion_query["context_hints"] = {
            "recent_intent": state.recent_intent,
            "recent_destination": state.recent_destination,
            "recent_occasion": state.recent_occasion,
            "recent_gender": state.recent_gender,
        }
        state.fashion_query = fashion_query
        state.ambiguities = fashion_query.get("disambiguation", [])
        
        query_type = fashion_query.get("query_type")
        self.logger.info(f"[PARSE] Classified query_type='{query_type}' | full_query={fashion_query}")
        
        if query_type == "broad":
            state.broad_intent = fashion_query
            self.logger.warning(f"[PARSE] Broad query detected - will trigger web_search | query={fashion_query}")
        
        state.log_event("parse_node", {"fashion_query": fashion_query})
        return state

