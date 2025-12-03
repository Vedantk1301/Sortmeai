"""
MergeNode combines validated catalog and web products into the final ranked set.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..state import MuseState


class MergeNode:
    def __call__(self, state: MuseState) -> MuseState:
        merged = list(state.qdrant_valid) + list(state.web_valid)
        merged.sort(key=lambda item: item.get("validator_score", 0), reverse=True)
        state.final_products = merged[:8]
        state.log_event("merge_node", {"final_count": len(state.final_products)})
        return state
