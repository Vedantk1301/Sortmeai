"""
KnowledgePlannerAgent expands broad travel/event intents into product and research queries.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from config import Config
from services.llm import LLM

PLANNER_PROMPT = """You are a fashion planning assistant.
Given a broad intent (travel/event/seasonal/occasion), produce:

**If destination is mentioned (travel/location query):**
- weather_search_query: text to ask for weather context at destination
- web_queries: 2-3 web queries to learn cultural norms and trending outfits for that location
- product_queries: 3-4 catalog-ready product micro-queries for retrieval (MAX 4!)

**If NO destination (general queries like "summer outfits", "office wear", "party outfits"):**
- weather_search_query: null (or omit)
- web_queries: [] (empty - we don't need web search for general fashion queries)
- product_queries: 3-4 catalog-ready product micro-queries covering different categories (MAX 4!)

CRITICAL: Generate MAXIMUM 4 product_queries for speed. Quality over quantity!

Rules:
- Tailor queries to gender if provided.
- Include fabric hints for seasons (linen/cotton for summer, wool/fleece for winter).
- For occasions, vary styles (casual, formal, trendy, classic).
- Output strict JSON with keys: weather_search_query, web_queries, product_queries.
"""


class KnowledgePlannerAgent:
    def __init__(self, llm: LLM | None = None, ledger_hook=None) -> None:
        self.llm = llm or LLM()
        self.ledger_hook = ledger_hook

    def __call__(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger(__name__)
        
        user_payload = {
            "destination": intent.get("destination"),
            "occasion": intent.get("occasion"),
            "gender": intent.get("gender"),
            "raw_query": intent.get("raw_query"),
            "context_hints": intent.get("context_hints") or {},
        }
        
        logger.info(f"[PLANNER] Input intent: {user_payload}")
        
        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        content = self.llm.chat(
            model=Config.FAST_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        plan = json.loads(content)
        
        # Only set default weather query if there's a destination
        destination = intent.get('destination')
        if destination and not plan.get('weather_search_query'):
            plan['weather_search_query'] = f"current weather and next 7 days forecast in {destination}"
        
        plan.setdefault("web_queries", [])
        plan.setdefault("product_queries", [])

        # Enforce gender hints to avoid mixing mens/womens when we already know the preference.
        plan = self._enforce_gender(plan, intent.get("gender"))
        
        logger.info(f"[PLANNER] Generated plan:")
        logger.info(f"  - destination: {destination or 'None'}")
        logger.info(f"  - weather_search_query: {plan.get('weather_search_query', 'None')}")
        logger.info(f"  - web_queries: {len(plan.get('web_queries', []))} queries")
        logger.info(f"  - product_queries: {len(plan.get('product_queries', []))} queries")
        if plan.get('web_queries'):
            logger.warning(f"[PLANNER] Web search will be triggered: {plan['web_queries']}")
        
        if self.ledger_hook:
            self.ledger_hook({"intent": user_payload, "plan": plan}, component="knowledge_planner")
        return plan

    def _enforce_gender(self, plan: Dict[str, Any], gender: Any) -> Dict[str, Any]:
        if not gender:
            return plan
        g = str(gender).lower()
        if g not in ("men", "women"):
            return plan  # unisex/both/unknown - leave as-is

        target_prefix = "men's" if g == "men" else "women's"
        other_token = "women" if g == "men" else "men"

        fixed_queries: List[str] = []
        for q in plan.get("product_queries", []):
            text = str(q).strip()
            lower = text.lower()
            # If it already matches target gender, keep
            if target_prefix.split("'")[0] in lower or g in lower:
                fixed_queries.append(text)
                continue
            # If it mentions the opposite gender, replace it
            if other_token in lower:
                text = lower.replace(other_token, target_prefix)
            else:
                text = f"{target_prefix} {text}"
            fixed_queries.append(text)

        plan["product_queries"] = fixed_queries
        return plan
