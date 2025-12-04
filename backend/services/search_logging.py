"""
Helpers for structured, per-run product search logging.
Writes the latest search (single or multi-query) into a JSON file for quick debugging.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Always overwrite with the latest run to keep the file small and focused.
PRODUCT_SEARCH_LOG = Path(__file__).resolve().parents[1] / "cache" / "product_search_log.json"


def summarize_products(items: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    """Compact view of products for logging."""
    summary: List[Dict[str, Any]] = []
    for item in items[:limit]:
        price = item.get("price") or {}
        summary.append(
            {
                "id": str(item.get("id")),
                "title": item.get("title"),
                "brand": item.get("brand"),
                "price": price.get("current"),
                "currency": price.get("currency"),
                "score": item.get("score"),
                "source": item.get("source") or item.get("origin"),
                "url": item.get("url"),
            }
        )
    return summary


def write_product_search_log(payload: Dict[str, Any]) -> None:
    """
    Persist the latest search run to a JSON file (overwrites on every call).
    Useful for inspecting what was sent to Qdrant, what came back, and how reranking behaved.
    """
    try:
        PRODUCT_SEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        PRODUCT_SEARCH_LOG.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive logging only
        logging.getLogger(__name__).warning("[SEARCH_LOG] Failed to write product search log: %s", exc)

