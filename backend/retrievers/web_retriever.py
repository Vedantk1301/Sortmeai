"""
WebRetriever uses the Responses API web_search tool as a fallback source.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from retrievers.indian_fashion_search import search_fashion_with_web
from config import Config


class WebRetriever:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def search(self, query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            raw = search_fashion_with_web(query_text, max_results=limit)
            return [self._normalise(idx, item) for idx, item in enumerate(raw)]
        except Exception as exc:
            self.logger.warning("Web retriever failed; returning empty list", exc_info=exc)
            return []

    def _normalise(self, idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": f"web-{idx}",
            "title": item.get("name") or f"Web result {idx}",
            "brand": "WebSource",
            "price": {"value": item.get("price"), "currency": "INR"},
            "image_url": item.get("imageUrl"),
            "url": item.get("sourceUrl"),
            "color": [],
            "pattern": None,
            "fit": None,
            "fabric": None,
            "gender": None,
            "tags": [item.get("tone")] if item.get("tone") else [],
            "attributes": {"source": "web"},
            "source": "web",
            "score": 0.5,
        }
