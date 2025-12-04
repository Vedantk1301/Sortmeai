"""
MergeNode combines validated catalog and web products into the final ranked set.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..state import SortmeState


class MergeNode:
    def __call__(self, state: SortmeState) -> SortmeState:
        merged = list(state.qdrant_valid) + list(state.web_valid)
        merged.sort(key=lambda item: item.get("validator_score", 0), reverse=True)
        
        # Deduplicate by ID
        seen_ids = set()
        deduped = []
        for item in merged:
            item_id = item.get("id")
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                deduped.append(item)
        
        state.final_products = deduped[:8]
        state.log_event("merge_node", {"final_count": len(state.final_products)})
        return state

