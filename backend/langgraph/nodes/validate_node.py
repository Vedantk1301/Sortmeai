"""
Validation nodes apply vision checks to catalog and web candidates.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from validators import VisionValidator
from ..state import SortmeState


class VisionValidateNode:
    def __init__(self, validator: VisionValidator | None = None) -> None:
        self.validator = validator or VisionValidator()

    def __call__(self, state: SortmeState) -> SortmeState:
        candidates = state.qdrant_filtered or state.qdrant_candidates
        if not candidates:
            return state

        result = self.validator.validate(state.fashion_query or {}, candidates, source="qdrant")
        state.qdrant_valid = self._attach_products(candidates, result["valid"])
        state.log_event("vision_validate_node", {"valid": len(state.qdrant_valid), "invalid": len(result["invalid"])})
        return state

    def _attach_products(self, candidates: List[Dict], validated: List[Dict]) -> List[Dict]:
        index = {item.get("id"): item for item in candidates}
        merged = []
        for row in validated:
            product = deepcopy(index.get(row["id"], {"id": row["id"]}))
            product.update(
                {
                    "validator_score": row.get("score"),
                    "validator_tag": row.get("tag"),
                    "validator_reason": row.get("reason", ""),
                    "origin": product.get("source") or product.get("origin"),
                }
            )
            merged.append(product)
        return merged


class WebVisionValidateNode:
    def __init__(self, validator: VisionValidator | None = None) -> None:
        self.validator = validator or VisionValidator()

    def __call__(self, state: SortmeState) -> SortmeState:
        if not state.web_candidates:
            return state

        result = self.validator.validate(state.fashion_query or {}, state.web_candidates, source="web")
        state.web_valid = self._attach_products(state.web_candidates, result["valid"])
        state.log_event("web_vision_validate_node", {"valid": len(state.web_valid), "invalid": len(result["invalid"])})
        return state

    def _attach_products(self, candidates: List[Dict], validated: List[Dict]) -> List[Dict]:
        index = {item.get("id"): item for item in candidates}
        merged = []
        for row in validated:
            product = deepcopy(index.get(row["id"], {"id": row["id"]}))
            product.update(
                {
                    "validator_score": row.get("score"),
                    "validator_tag": row.get("tag"),
                    "validator_reason": row.get("reason", ""),
                    "origin": product.get("source") or product.get("origin"),
                }
            )
            merged.append(product)
        return merged

