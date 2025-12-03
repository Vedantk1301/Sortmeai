"""
ParseNode converts raw user text into the structured fashion_query.
"""

from __future__ import annotations

import logging
from agents import ParserAgent
from ..state import MuseState


class ParseNode:
    def __init__(self, parser: ParserAgent | None = None) -> None:
        self.parser = parser or ParserAgent()
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: MuseState) -> MuseState:
        self.logger.info(f"[PARSE] Processing user message: '{state.user_message}'")
        fashion_query = self.parser(state.user_message)
        state.fashion_query = fashion_query
        state.ambiguities = fashion_query.get("disambiguation", [])
        
        query_type = fashion_query.get("query_type")
        self.logger.info(f"[PARSE] Classified query_type='{query_type}' | full_query={fashion_query}")
        
        if query_type == "broad":
            state.broad_intent = fashion_query
            self.logger.warning(f"[PARSE] Broad query detected - will trigger web_search | query={fashion_query}")
        
        state.log_event("parse_node", {"fashion_query": fashion_query})
        return state
