"""
Retrieval nodes for catalog (Qdrant) and web search.
"""

from __future__ import annotations

import time
from retrievers import CatalogRetriever, Reranker, WebRetriever
from services.search_logging import summarize_products, write_product_search_log
from ..state import SortmeState


class CatalogRetrieveNode:
    def __init__(self, catalog: CatalogRetriever | None = None, reranker: Reranker | None = None) -> None:
        self.catalog = catalog or CatalogRetriever()
        self.reranker = reranker or Reranker()

    def __call__(self, state: SortmeState) -> SortmeState:
        if not state.fashion_query:
            return state

        import logging
        logger = logging.getLogger(__name__)
        trace_id = f"catalog-{int(time.time() * 1000)}"

        # 1. Catalog Search
        start_time = time.time()
        raw, search_debug = self.catalog.search(
            state.fashion_query, top_k=40, trace_id=trace_id, capture_debug=True
        )
        search_time = time.time() - start_time
        logger.info(f"[RETRIEVE] Catalog search returned {len(raw)} candidates | time={search_time:.3f}s")
        
        # 2. Reranking
        rerank_start = time.time()
        filtered, rerank_debug = self.reranker.rerank(
            state.fashion_query, raw, top_k=12, trace_id=trace_id, capture_debug=True
        )
        rerank_time = time.time() - rerank_start
        logger.info(f"[RETRIEVE] Reranking reduced {len(raw)} -> {len(filtered)} candidates | time={rerank_time:.3f}s")

        if filtered:
            top_score = filtered[0].get("score", 0)
            logger.info(f"[RETRIEVE] Top candidate score: {top_score:.4f}")

        write_product_search_log(
            {
                "trace_id": trace_id,
                "mode": "single",
                "qdrant_query": state.fashion_query,
                "query_text": search_debug.get("query_text"),
                "retrieval": {
                    "requested_top_k": search_debug.get("requested_top_k"),
                    "filters": search_debug.get("filters"),
                    "raw_preview": search_debug.get("raw_preview"),
                    "post_filter_preview": search_debug.get("post_filter_preview"),
                    "counts": {
                        "retrieved": search_debug.get("retrieved_count", len(raw)),
                        "returned": search_debug.get("returned_count", len(raw)),
                    },
                    "timing": search_debug.get("timing"),
                },
                "rerank": rerank_debug,
                "final_preview": summarize_products(filtered, limit=12),
            }
        )

        state.qdrant_candidates = raw
        state.qdrant_filtered = filtered
        state.log_event("catalog_retrieve_node", {"retrieved": len(raw), "filtered": len(filtered)})
        return state


class WebRetrieveNode:
    def __init__(self, retriever: WebRetriever | None = None) -> None:
        self.retriever = retriever or WebRetriever()

    def __call__(self, state: SortmeState) -> SortmeState:
        query_text = state.user_message
        state.web_candidates = self.retriever.search(query_text, limit=25)
        state.log_event("web_retrieve_node", {"retrieved": len(state.web_candidates)})
        return state

