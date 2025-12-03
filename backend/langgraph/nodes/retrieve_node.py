"""
Retrieval nodes for catalog (Qdrant) and web search.
"""

from __future__ import annotations

from retrievers import CatalogRetriever, Reranker, WebRetriever
from ..state import MuseState


class CatalogRetrieveNode:
    def __init__(self, catalog: CatalogRetriever | None = None, reranker: Reranker | None = None) -> None:
        self.catalog = catalog or CatalogRetriever()
        self.reranker = reranker or Reranker()

    def __call__(self, state: MuseState) -> MuseState:
        if not state.fashion_query:
            return state

        import time
        import logging
        logger = logging.getLogger(__name__)

        # 1. Catalog Search
        start_time = time.time()
        raw = self.catalog.search(state.fashion_query, top_k=40)
        search_time = time.time() - start_time
        logger.info(f"[RETRIEVE] Catalog search returned {len(raw)} candidates | time={search_time:.3f}s")
        
        # 2. Reranking
        rerank_start = time.time()
        filtered = self.reranker.rerank(state.fashion_query, raw, top_k=12)
        rerank_time = time.time() - rerank_start
        logger.info(f"[RETRIEVE] Reranking reduced {len(raw)} -> {len(filtered)} candidates | time={rerank_time:.3f}s")

        if filtered:
            top_score = filtered[0].get("score", 0)
            logger.info(f"[RETRIEVE] Top candidate score: {top_score:.4f}")

        state.qdrant_candidates = raw
        state.qdrant_filtered = filtered
        state.log_event("catalog_retrieve_node", {"retrieved": len(raw), "filtered": len(filtered)})
        return state


class WebRetrieveNode:
    def __init__(self, retriever: WebRetriever | None = None) -> None:
        self.retriever = retriever or WebRetriever()

    def __call__(self, state: MuseState) -> MuseState:
        query_text = state.user_message
        state.web_candidates = self.retriever.search(query_text, limit=25)
        state.log_event("web_retrieve_node", {"retrieved": len(state.web_candidates)})
        return state
