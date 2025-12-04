"""
Reranker trims the raw catalog results using DeepInfra Qwen reranker.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from services.deepinfra import RERANK_MODEL, rerank_qwen_sync
from services.search_logging import summarize_products


class Reranker:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def rerank(
        self,
        query: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_k: int = 12,
        trace_id: str | None = None,
        capture_debug: bool = False,
    ) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], Dict[str, Any]]:
        import time
        trace_label = trace_id or f"rerank-{int(time.time() * 1000)}"
        query_text = self._render_query(query)
        documents = [self._render_doc(c) for c in candidates]
        debug: Dict[str, Any] = {
            "trace_id": trace_label,
            "model": RERANK_MODEL,
            "query_text": query_text,
            "requested_top_k": top_k,
        }
        if not candidates:
            debug["note"] = "no candidates to rerank"
            return ([], debug) if capture_debug else []

        try:
            order = rerank_qwen_sync(query_text, documents, top_k=top_k)
            debug["order"] = order
        except Exception as exc:
            debug["error"] = str(exc)
            order = list(range(min(top_k, len(candidates))))
        
        # Get reranked candidates based on order
        reranked_candidates = [candidates[i] for i in order if i < len(candidates)]
        
        # Apply brand diversity only to top 20 to avoid promoting very low relevance items
        diversity_scope = 20
        top_candidates = reranked_candidates[:diversity_scope]
        rest_candidates = reranked_candidates[diversity_scope:]
        
        balanced_top = self._apply_brand_diversity(top_candidates, max_per_brand=2)
        balanced_candidates = balanced_top + rest_candidates
        
        ranked = balanced_candidates[:top_k]
        preview = summarize_products(ranked, limit=top_k)
        debug["top_preview"] = preview
        self.logger.info(
            f"[RERANK][{trace_label}] {len(candidates)} -> {len(ranked)} using {RERANK_MODEL} | diversity_applied=True"
        )
        self.logger.info(f"[RERANK][{trace_label}] Top reranked: {preview}")
        return (ranked, debug) if capture_debug else ranked

    def _apply_brand_diversity(self, candidates: List[Dict[str, Any]], max_per_brand: int = 2) -> List[Dict[str, Any]]:
        """Reorder candidates to ensure brand diversity in top results."""
        brand_counts: Dict[str, int] = {}
        diverse: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        
        for item in candidates:
            brand = (item.get("brand") or "unknown").lower()
            if brand_counts.get(brand, 0) < max_per_brand:
                diverse.append(item)
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            else:
                deferred.append(item)
        
        # Append deferred items at the end
        return diverse + deferred

    def _render_query(self, query: Dict[str, Any]) -> str:
        if not isinstance(query, dict):
            return str(query)
        parts = [query.get("raw_query") or query.get("item_type") or ""]
        for key in ("colors", "pattern", "fit", "fabric", "occasion"):
            val = query.get(key)
            if not val:
                continue
            if isinstance(val, list):
                parts.append(" ".join(val))
            else:
                parts.append(str(val))
        return " ".join([p for p in parts if p]).strip()

    def _render_doc(self, item: Dict[str, Any]) -> str:
        # Strip brand to avoid bias
        brand = (item.get("brand") or "").lower()
        title = (item.get("title") or "").lower()
        
        if brand and brand in title:
            title = title.replace(brand, "").strip()
            
        fields = [
            title,
            # item.get("brand"), # Exclude brand
            item.get("pattern"),
            item.get("fabric"),
            str(item.get("price")),
            item.get("attributes"),
        ]
        return " ".join([str(f) for f in fields if f])
