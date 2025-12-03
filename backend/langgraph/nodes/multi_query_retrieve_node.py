"""
Runs multiple product micro-queries IN PARALLEL, then reranks once at the end.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List

from config import Config
from retrievers import CatalogRetriever, Reranker
from ..state import MuseState


class MultiQueryRetrieveNode:
    def __init__(
        self,
        catalog: CatalogRetriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.catalog = catalog or CatalogRetriever()
        self.reranker = reranker or Reranker()

    def __call__(self, state: MuseState) -> MuseState:
        logger = logging.getLogger(__name__)
        product_queries = (state.planner_plan or {}).get("product_queries") or []
        
        if not product_queries:
            return state
        
        # CRITICAL: Limit to max 4 queries for speed (with parallelism, 4 is manageable)
        product_queries = product_queries[:4]
        logger.info(f"[MULTI_QUERY] Running {len(product_queries)} queries IN PARALLEL")
        
        # Run all Qdrant searches in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_query = {
                executor.submit(self._search_single, state, pq): pq 
                for pq in product_queries
            }
            
            for future in as_completed(future_to_query):
                pq = future_to_query[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"[MULTI_QUERY] Query '{pq[:50]}' returned {len(results)} results")
                except Exception as e:
                    logger.error(f"[MULTI_QUERY] Query '{pq[:50]}' failed: {e}")
        
        # Dedupe
        all_results = self._dedupe(all_results)
        logger.info(f"[MULTI_QUERY] Total unique results after dedup: {len(all_results)}")
        
        # Single reranking pass at the end (much faster than per-query)
        if all_results:
            combined_query = " ".join(product_queries[:2])  # Use first 2 queries for context
            ranked = self.reranker.rerank(combined_query, all_results, top_k=12)
            logger.info(f"[MULTI_QUERY] Reranked to top {len(ranked)} products")
        else:
            ranked = []
        
        # Skip vision validation for broad queries (WAY too slow - 30s per batch!)
        # Vision validation should only run for specific product searches
        state.pooled_valid_products = ranked
        state.final_products = ranked[:8]
        
        logger.info(f"[MULTI_QUERY] Returning {len(state.final_products)} products")
        state.log_event("multi_query_retrieve_node", {"pooled": len(ranked)})
        return state
    
    def _search_single(self, state: MuseState, query_text: str) -> List[Dict[str, Any]]:
        """Search for a single query"""
        sub_query = self._build_subquery(state, query_text)
        results = self.catalog.search(sub_query, top_k=40)  # Get more results per query
        return results

    def _build_subquery(self, state: MuseState, query_text: str) -> Dict[str, Any]:
        base = deepcopy(state.fashion_query or {})
        base["item_type"] = query_text
        base["raw_query"] = query_text
        if state.interpretation_flags:
            base["interpretation"] = state.interpretation_flags
        return base

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
