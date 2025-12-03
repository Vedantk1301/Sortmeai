"""
WebFashionNode mines dressing rules from planner web queries.
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import Config
from services.web_search import WebSearchClient
from ..state import SortmeState


class WebFashionNode:
    def __init__(self, web_client: WebSearchClient | None = None) -> None:
        self.web_client = web_client or WebSearchClient()

    def __call__(self, state: SortmeState) -> SortmeState:
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        queries = (state.planner_plan or {}).get("web_queries") or []
        logger.info(f"[WEB_SEARCH] web_queries={queries}")
        
        if not queries:
            logger.info("[WEB_SEARCH] No web queries, skipping")
            return state
        
        start_time = time.time()
        rules: List[str] = []
        for q in queries:
            logger.info(f"[WEB_SEARCH] Executing query: '{q}'")
            query_start = time.time()
            results = self.web_client.search(q, max_results=5)
            query_time = time.time() - query_start
            logger.info(f"[WEB_SEARCH] Query completed | time={query_time:.3f}s | results={len(results)}")
            rules.extend(self.web_client.extract_rules(results))
        
        deduped = self._dedupe(rules)
        total_time = time.time() - start_time
        logger.info(f"[WEB_SEARCH] Complete | total_time={total_time:.3f}s | rules={len(deduped)}")
        
        state.fashion_knowledge = {"destination": (state.fashion_query or {}).get("destination"), "rules": deduped}
        state.log_event("web_fashion_node", {"rule_count": len(deduped)})
        return state

    def _dedupe(self, rules: List[str]) -> List[str]:
        seen = set()
        out = []
        for r in rules:
            if r and r not in seen:
                seen.add(r)
                out.append(r)
        return out

