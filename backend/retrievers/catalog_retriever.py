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
from services.search_logging import summarize_products


class CatalogRetriever:
    def __init__(self, client=None) -> None:
        self.client = client or get_qdrant_client()
        self.logger = logging.getLogger(__name__)

    def search(
        self,
        query: Dict[str, Any],
        top_k: int = Config.SEARCH_LIMIT,
        trace_id: str | None = None,
        capture_debug: bool = False,
    ) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], Dict[str, Any]]:
        import time

        trace_label = trace_id or f"qdrant-{int(time.time() * 1000)}"
        start_time = time.time()
        query_text = self._to_query_text(query)
        filter_summary = self._filter_summary(query)
        debug: Dict[str, Any] = {
            "trace_id": trace_label,
            "query_text": query_text,
            "filters": filter_summary,
            "collection": Config.CATALOG_COLLECTION,
            "requested_top_k": top_k,
        }
        self.logger.info(
            f"[QDRANT][{trace_label}] Starting search | query_text='{query_text}' | "
            f"top_k={top_k} | collection={Config.CATALOG_COLLECTION} | filters={filter_summary}"
        )
        
        try:
            # Embedding
            embed_start = time.time()
            vector = embed_catalog_sync([query_text])[0]
            embed_time = time.time() - embed_start
            debug["timing"] = {"embed": embed_time}
            self.logger.info(f"[QDRANT][{trace_label}] Embedding completed | time={embed_time:.3f}s")
            
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
            debug["timing"]["search"] = search_time
            debug["retrieved_count"] = num_results
            self.logger.info(
                f"[QDRANT][{trace_label}] Vector search completed | results={num_results} | time={search_time:.3f}s"
            )
            
            raw_products = [self._to_product(point, query) for point in (result.points or [])]
            debug["raw_preview"] = summarize_products(raw_products, limit=8)
            
            # 1. Junk Filter
            products = [p for p in raw_products if not self._is_disallowed_product(p)]
            
            # 2. Brand Cap
            products = self._rebalance_brand_pool(products)
            
            # Trim to requested top_k
            products = products[:top_k]
            debug["post_filter_preview"] = summarize_products(products, limit=8)
            debug["returned_count"] = len(products)

            total_time = time.time() - start_time
            debug["total_time"] = total_time
            self.logger.info(
                f"[QDRANT][{trace_label}] Search complete | total_time={total_time:.3f}s | "
                f"embed={embed_time:.3f}s | search={search_time:.3f}s | returned={len(products)}"
            )
            return (products, debug) if capture_debug else products
        except Exception as exc:
            debug["error"] = str(exc)
            self.logger.warning(
                f"[QDRANT][{trace_label}] Search fallback to heuristic (embedding error) | error={str(exc)}",
                exc_info=exc,
            )
            fallback = [self._mock_product(idx, query, source="qdrant-fallback") for idx in range(min(top_k, 8))]
            return (fallback, debug) if capture_debug else fallback

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
        
        # Extract pricing info
        # Check if payload has a price dict (shouldn't happen, but be defensive)
        payload_price = payload.get("price")
        if isinstance(payload_price, dict):
            # Price is already a dict - extract current value from it
            current_price = payload_price.get("current") or payload_price.get("value") or attrs.get("price_inr")
        else:
            # Price is a number or None
            current_price = attrs.get("price_inr") or payload_price or None
        
        mrp = attrs.get("mrp") or attrs.get("compare_at_price") or payload.get("mrp") or None
        
        # Calculate discount percentage
        discount_pct = 0
        if current_price and mrp and mrp > current_price:
            discount_pct = ((mrp - current_price) / mrp) * 100
        
        # Build price object matching frontend expectations
        # Ensure current is a number (0 if None) to avoid React rendering errors
        price_obj = {
            "current": current_price if current_price is not None else 0,
            "currency": "₹",
            "compare_at": mrp if mrp and mrp > (current_price or 0) else None,
            "discount_pct": discount_pct if discount_pct > 0 else None,
        }
        
        colors = payload.get("color") or payload.get("colors") or attrs.get("color") or []
        if isinstance(colors, str):
            colors = [colors]
        return {
            "id": str(payload.get("product_id") or payload.get("id") or point.id),
            "title": payload.get("title") or attrs.get("title") or query.get("item_type", "product"),
            "brand": payload.get("brand") or attrs.get("brand"),
            "price": price_obj,
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
            "price": {
                "current": 0,  # Fallback products don't have real prices
                "currency": "₹",
                "compare_at": None,
                "discount_pct": None,
            },
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

    def _filter_summary(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight, serializable view of filters applied to Qdrant."""
        gender = query.get("gender")
        min_price = query.get("min_price")
        max_price = query.get("max_price")
        summary: Dict[str, Any] = {}
        if gender:
            # Unisex is allowed automatically for men/women in _gender_filter
            summary["gender"] = gender
        if min_price is not None or max_price is not None:
            summary["price_inr"] = {"min": min_price, "max": max_price}
        return summary
