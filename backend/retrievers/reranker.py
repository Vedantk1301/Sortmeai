"""
Reranker trims the raw catalog results using DeepInfra Qwen reranker.
"""

from __future__ import annotations

from typing import Any, Dict, List

from services.deepinfra import rerank_qwen_sync


class Reranker:
    def rerank(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], top_k: int = 12) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        query_text = self._render_query(query)
        documents = [self._render_doc(c) for c in candidates]
        try:
            order = rerank_qwen_sync(query_text, documents, top_k=top_k)
        except Exception:
            order = list(range(min(top_k, len(candidates))))
        ranked = [candidates[i] for i in order if i < len(candidates)]
        return ranked[:top_k]

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
