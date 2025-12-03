"""
CatalogRetriever handles Qdrant vector search with OpenAI embeddings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from qdrant_client.http import models as rest

from config import Config
from services.deepinfra import embed_catalog_sync
from services.qdrant_client import get_qdrant_client


class CatalogRetriever:
    def __init__(self, client=None) -> None:
        self.client = client or get_qdrant_client()
        self.logger = logging.getLogger(__name__)

    def search(self, query: Dict[str, Any], top_k: int = Config.SEARCH_LIMIT) -> List[Dict[str, Any]]:
        import time
        start_time = time.time()
        query_text = self._to_query_text(query)
        self.logger.info(f"[QDRANT] Starting search | query_text='{query_text}' | top_k={top_k} | collection={Config.CATALOG_COLLECTION}")
        
        try:
            # Embedding
            embed_start = time.time()
            vector = embed_catalog_sync([query_text])[0]
            embed_time = time.time() - embed_start
            self.logger.info(f"[QDRANT] Embedding completed | time={embed_time:.3f}s")
            
            # Build filters - ONLY gender and price
            filters = []
            gender_f = self._gender_filter(query.get("gender"))
            if gender_f:
                filters.append(gender_f)
            
            price_f = self._price_filter(query.get("min_price"), query.get("max_price"))
            if price_f:
                filters.append(price_f)

            final_filter = rest.Filter(must=filters) if filters else None

            # Vector search
            search_start = time.time()
            result = self.client.query_points(
                collection_name=Config.CATALOG_COLLECTION,
                query=vector,
                limit=top_k * 2, # Fetch more for filtering
                with_payload=True,
                search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF),
                query_filter=final_filter,
            )
            search_time = time.time() - search_start
            num_results = len(result.points or [])
            self.logger.info(f"[QDRANT] Vector search completed | results={num_results} | time={search_time:.3f}s")
            
            products = [self._to_product(point, query) for point in (result.points or [])]
            
            # 1. Junk Filter
            products = [p for p in products if not self._is_disallowed_product(p)]
            
            # 2. Brand Cap
            products = self._rebalance_brand_pool(products)
            
            # Trim to requested top_k
            products = products[:top_k]

            total_time = time.time() - start_time
            self.logger.info(f"[QDRANT] Search complete | total_time={total_time:.3f}s | embed={embed_time:.3f}s | search={search_time:.3f}s")
            return products
        except Exception as exc:
            self.logger.warning(f"[QDRANT] Search fallback to heuristic (embedding error) | error={str(exc)}", exc_info=exc)
            return [self._mock_product(idx, query, source="qdrant-fallback") for idx in range(min(top_k, 8))]

    def _to_query_text(self, query: Dict[str, Any]) -> str:
        # Build query text from raw_query or item_type only
        # DO NOT include colors, pattern, fabric - let vector search handle it
        parts = []
        
        raw = query.get("raw_query")
        if raw:
            parts.append(raw)
        else:
            item_type = query.get("item_type")
            if item_type:
                parts.append(item_type)
        
        # Include destination/occasion for context if present
        for key in ("destination", "occasion"):
            if query.get(key):
                parts.append(str(query[key]))
        
        return " ".join(parts) if parts else "fashion item"

    def _gender_filter(self, gender: str | None):
        if not gender or gender == "unknown":
            return None
        allowed = [gender]
        if gender in ("men", "women"):
            allowed.append("unisex")
        return rest.FieldCondition(
            key="attributes.gender",
            match=rest.MatchAny(any=allowed),
        )

    def _price_filter(self, min_p: float | None, max_p: float | None):
        if min_p is None and max_p is None:
            return None
        
        r = rest.Range(gte=min_p, lte=max_p)
        return rest.FieldCondition(key="attributes.price_inr", range=r)

    def _is_disallowed_product(self, product: Dict[str, Any]) -> bool:
        blocked = {
            "socks", "sock", "hosiery", "stocking", "brief", "briefs",
            "panty", "panties", "lingerie", "innerwear", "underwear",
            "undergarment", "bra", "camisole", "thermal", "thermals",
            "mask", "muffler"
        }
        text_parts = [
            str(product.get("category") or ""),
            str(product.get("title") or ""),
        ]
        text = " ".join(text_parts).lower()
        return any(tok in text for tok in blocked)

    def _rebalance_brand_pool(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Simple cap: max 3 per brand in top results
        from collections import defaultdict
        cap = 3
        capped = []
        overflow = []
        counts = defaultdict(int)
        
        for p in products:
            brand = (p.get("brand") or "unknown").lower()
            if counts[brand] < cap:
                counts[brand] += 1
                capped.append(p)
            else:
                overflow.append(p)
        
        return capped + overflow

    def _to_product(self, point: Any, query: Dict[str, Any]) -> Dict[str, Any]:
        payload = point.payload or {}
        attrs = payload.get("attributes") or {}
        price_val = attrs.get("price_inr") or payload.get("price") or None
        colors = payload.get("color") or payload.get("colors") or attrs.get("color") or []
        if isinstance(colors, str):
            colors = [colors]
        return {
            "id": str(payload.get("product_id") or payload.get("id") or point.id),
            "title": payload.get("title") or attrs.get("title") or query.get("item_type", "product"),
            "brand": payload.get("brand") or attrs.get("brand"),
            "price": {"value": price_val, "currency": "INR"},
            "image_url": payload.get("primary_image") or payload.get("image_url"),
            "url": payload.get("url"),
            "color": colors,
            "pattern": payload.get("pattern") or attrs.get("pattern"),
            "fit": payload.get("fit") or attrs.get("fit"),
            "fabric": payload.get("fabric") or attrs.get("fabric"),
            "gender": attrs.get("gender"),
            "tags": payload.get("tags") or [],
            "attributes": attrs,
            "source": "qdrant",
            "score": float(getattr(point, "score", 0.0)),
        }

    def _mock_product(self, idx: int, query: Dict[str, Any], source: str) -> Dict[str, Any]:
        color = query.get("colors", ["orange"])
        title_color = " ".join(color) if isinstance(color, list) else color
        item_type = query.get("item_type", "item")
        return {
            "id": f"{source}-{idx}",
            "title": f"{title_color.title()} {item_type.title()} {idx}",
            "brand": "Fallback",
            "price": {"value": None, "currency": None},
            "image_url": f"https://example.com/{source}-{idx}.jpg",
            "url": None,
            "color": color,
            "pattern": query.get("pattern"),
            "fit": query.get("fit"),
            "fabric": query.get("fabric"),
            "gender": query.get("gender"),
            "tags": [],
            "attributes": {"source_rank": idx},
            "source": source,
            "score": 1.0 - (idx * 0.01),
        }
