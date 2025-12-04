"""
Runs multiple product micro-queries IN PARALLEL, then reranks once at the end.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List

from config import Config
from retrievers import CatalogRetriever, Reranker
from services.search_logging import summarize_products, write_product_search_log
from ..state import SortmeState


class MultiQueryRetrieveNode:
    def __init__(
        self,
        catalog: CatalogRetriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.catalog = catalog or CatalogRetriever()
        self.reranker = reranker or Reranker()

    def __call__(self, state: SortmeState) -> SortmeState:
        logger = logging.getLogger(__name__)
        product_queries = (state.planner_plan or {}).get("product_queries") or []
        trace_id = f"multi-query-{int(time.time() * 1000)}"
        
        if not product_queries:
            return state
        
        # CRITICAL: Limit to max 4 queries for speed (with parallelism, 4 is manageable)
        # Filter out shoe/footwear queries as we don't carry them
        forbidden = ["shoe", "sneaker", "boot", "sandal", "footwear", "slipper", "heel", "flat"]
        product_queries = [
            q for q in product_queries 
            if not any(bad in q.lower() for bad in forbidden)
        ]
        
        product_queries = product_queries[:4]
        logger.info(f"[MULTI_QUERY] Running {len(product_queries)} queries IN PARALLEL")
        
        # Run all Qdrant searches in parallel
        all_results: List[Dict[str, Any]] = []
        per_query_logs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_query = {
                executor.submit(self._search_single, state, pq, f"{trace_id}-q{idx}"): pq 
                for idx, pq in enumerate(product_queries)
            }
            
            for future in as_completed(future_to_query):
                pq = future_to_query[future]
                try:
                    results, search_debug = future.result()
                    all_results.extend(results)
                    per_query_logs.append(
                        {
                            "trace_id": search_debug.get("trace_id"),
                            "query_text": search_debug.get("query_text") or pq,
                            "filters": search_debug.get("filters"),
                            "counts": {
                                "retrieved": search_debug.get("retrieved_count", len(results)),
                                "returned": search_debug.get("returned_count", len(results)),
                            },
                            "timing": search_debug.get("timing"),
                            "preview": search_debug.get("post_filter_preview")
                            or summarize_products(results, limit=6),
                        }
                    )
                    logger.info(f"[MULTI_QUERY] Query '{pq[:50]}' returned {len(results)} results")
                except Exception as e:
                    logger.error(f"[MULTI_QUERY] Query '{pq[:50]}' failed: {e}")
        
        # Dedupe
        total_collected = len(all_results)
        all_results = self._dedupe(all_results)
        logger.info(f"[MULTI_QUERY] Total unique results after dedup: {len(all_results)} (from {total_collected})")
        
        # Single reranking pass at the end (much faster than per-query)
        if all_results:
            combined_query = " | ".join(product_queries)
            ranked, rerank_debug = self.reranker.rerank(
                combined_query, all_results, top_k=40, trace_id=trace_id, capture_debug=True
            )
            balanced = self._balance_by_query(ranked, product_queries, per_query_cap=3)
            logger.info(f"[MULTI_QUERY] Reranked to top {len(ranked)} products | balanced to {len(balanced)}")
        else:
            ranked = []
            balanced = []
            rerank_debug = {"trace_id": trace_id, "note": "no results to rerank"}
        
        # Skip vision validation for broad queries (WAY too slow - 30s per batch!)
        # Vision validation should only run for specific product searches
        state.pooled_valid_products = balanced or ranked
        state.final_products = (balanced or ranked)[:8]
        
        write_product_search_log(
            {
                "trace_id": trace_id,
                "mode": "multi",
                "queries": product_queries,
                "per_query": per_query_logs,
                "counts": {"collected": total_collected, "deduped": len(all_results)},
                "rerank": rerank_debug,
                "final_preview": summarize_products(balanced or ranked, limit=12),
            }
        )

        logger.info(f"[MULTI_QUERY] Returning {len(state.final_products)} products")
        state.log_event("multi_query_retrieve_node", {"pooled": len(ranked)})
        return state
    
    def _search_single(
        self, state: SortmeState, query_text: str, trace_id: str
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for a single query"""
        sub_query = self._build_subquery(state, query_text)
        results, debug = self.catalog.search(
            sub_query, top_k=40, trace_id=trace_id, capture_debug=True
        )  # Get more results per query
        for item in results:
            item["origin_query"] = query_text
        return results, debug

    def _build_subquery(self, state: SortmeState, query_text: str) -> Dict[str, Any]:
        base = deepcopy(state.fashion_query or {})
        base["item_type"] = query_text
        base["raw_query"] = query_text
        if state.interpretation_flags:
            base["interpretation"] = state.interpretation_flags
        return base

    def _balance_by_query(
        self, ranked: List[Dict[str, Any]], queries: List[str], per_query_cap: int = 3
    ) -> List[Dict[str, Any]]:
        """Round-robin pick from reranked list to keep representation across queries."""
        buckets: Dict[str, List[Dict[str, Any]]] = {q: [] for q in queries}
        for item in ranked:
            origin = item.get("origin_query")
            if origin in buckets and len(buckets[origin]) < per_query_cap:
                buckets[origin].append(item)

        balanced: List[Dict[str, Any]] = []
        # Round-robin to keep mix
        while len(balanced) < len(ranked) and any(buckets.values()):
            for q in queries:
                bucket = buckets.get(q, [])
                if bucket:
                    balanced.append(bucket.pop(0))
                    if len(balanced) >= len(ranked):
                        break

        # Fill any remaining slots with the original reranked order to preserve strongest hits
        seen_ids = {id(item) for item in balanced}
        for item in ranked:
            if id(item) not in seen_ids:
                balanced.append(item)
        return balanced

    def _dedupe(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for p in products:
            pid = p.get("id") or p.get("url")
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append(p)
        return deduped

