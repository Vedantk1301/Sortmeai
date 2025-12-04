"""
Thin wrapper around OpenAI web_search for stylistic knowledge mining.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
import time

from openai import OpenAI

from config import Config
from .llm import LLM

RULE_PROMPT = """You are a fashion researcher. Given web snippets, extract 3-5 concise dressing rules or norms.
Output a JSON object: {"rules": ["rule1", "rule2", ...]}
Stay factual to snippets; do not invent."""


class WebSearchClient:
    def __init__(self, llm: LLM | None = None, cache_ttl: int = 600) -> None:
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.llm = llm or LLM()
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        now = time.time()
        if query in self._cache:
            ts, hits = self._cache[query]
            if now - ts < self.cache_ttl:
                return hits
        try:
            resp = self.client.responses.create(
                model=Config.WEATHER_MODEL,
                input=[{"role": "user", "content": query}],
                tools=[{"type": "web_search"}],  # modern Responses API shape
                tool_choice="auto",
            )
            hits: List[Dict[str, Any]] = []
            for item in resp.output or []:
                if item.type == "web_search_result":
                    hits.extend(item.content or [])
            hits = hits[:max_results]
            self._cache[query] = (now, hits)
            return hits
        except Exception as exc:  # graceful fallback instead of 500
            import logging
            logging.getLogger(__name__).warning(f"[WEB_SEARCH] Failed web search '{query}': {exc}")
            return []

    def extract_rules(self, search_results: List[Dict[str, Any]]) -> List[str]:
        if not search_results:
            return []
        payload = {"results": search_results}
        messages = [
            {"role": "system", "content": RULE_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        content = self.llm.chat(
            model=Config.WEATHER_MODEL,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(content)
        rules = data.get("rules") or []
        return [r for r in rules if isinstance(r, str)]
