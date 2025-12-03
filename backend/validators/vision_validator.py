"""
Vision validator backed by GPT-5 with a heuristic fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from config import Config
from services.llm import LLM

VISION_PROMPT = """You are a strict fashion product validator. 
Your job is to check whether each product matches the user's structured query.

You MUST:
- Use both the image and metadata
- Prioritize accuracy over inclusiveness
- Reject ANY product that does not strictly match the pattern, color combination, or category
- Explain why each rejected product failed, in one short sentence
- Assign ranking tags:
    "best_match", "close_match", "weak_match"
- Assign a score from 0.0 to 1.0

Output JSON only:
{
  "valid": [{ "id": "...", "score": 0.95, "tag": "best_match", "reason": "" }],
  "invalid": [{ "id": "...", "reason": "...", "tag": "weak_match" }]
}"""


class VisionValidator:
    def __init__(self, llm_client: LLM | None = None, ledger_hook=None) -> None:
        self.llm_client = llm_client or LLM()
        self.ledger_hook = ledger_hook
        self.logger = logging.getLogger(__name__)

    def validate(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], source: str = "qdrant") -> Dict[str, Any]:
        try:
            results = self._llm_validate(query, candidates, source=source)
        except Exception as exc:  # network or parsing failure
            self.logger.warning("Vision validator fell back to heuristic", exc_info=exc)
            results = self._heuristic_validate(query, candidates, source)

        if self.ledger_hook:
            self.ledger_hook({"prompt": VISION_PROMPT, "results": results}, component="vision_validator")
        return results

    def _llm_validate(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
        self.logger.info(f"[VISION] Validating {len(candidates)} candidates from {source}")
        
        payload = {"query": query, "candidates": candidates, "source": source}
        user_parts: List[Dict[str, Any]] = [
            {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}
        ]
        for cand in candidates:
            image_url = cand.get("image_url")
            if image_url:
                # Standard OpenAI format
                user_parts.append({"type": "image_url", "image_url": {"url": image_url}})

        inputs = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": VISION_PROMPT}],
            },
            {
                "role": "user",
                "content": user_parts,
            },
        ]

        content = self.llm_client.chat(
            model=Config.AGENT_MODEL,
            messages=inputs,
            response_format={"type": "json_object"},
        )
        data = json.loads(content)
        valid = data.get("valid") or []
        invalid = data.get("invalid") or []
        
        self.logger.info(f"[VISION] Result: {len(valid)} valid, {len(invalid)} invalid")
        if invalid:
            first_reason = invalid[0].get("reason", "unknown")
            self.logger.info(f"[VISION] Sample rejection: {first_reason}")

        enriched_valid = []
        for idx, item in enumerate(valid):
            item.setdefault("tag", "close_match")
            item.setdefault("score", round(0.85 - idx * 0.02, 2))
            item.setdefault("reason", "")
            item["is_valid"] = True
            enriched_valid.append(item)
        enriched_invalid = []
        for item in invalid:
            item["is_valid"] = False
            item.setdefault("tag", "weak_match")
            enriched_invalid.append(item)
        return {"valid": enriched_valid, "invalid": enriched_invalid}

    def _heuristic_validate(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
        results = {"valid": [], "invalid": []}
        for idx, product in enumerate(candidates):
            reason = self._validate_product(query, product)
            if reason:
                results["invalid"].append({"id": product.get("id"), "reason": reason, "tag": "weak_match", "is_valid": False})
            else:
                score = self._score_product(idx, source=source)
                tag = "best_match" if idx == 0 else "close_match"
                results["valid"].append({"id": product.get("id"), "score": score, "tag": tag, "reason": "", "is_valid": True})
        return results

    def _validate_product(self, query: Dict[str, Any], product: Dict[str, Any]) -> str:
        product_colors = product.get("color") or []
        if isinstance(product_colors, str):
            product_colors = [product_colors]
        product_colors = [p.lower() for p in product_colors]
        missing_colors = [c for c in query.get("colors", []) if c not in product_colors]
        if missing_colors and query.get("color_mode") == "all_required":
            return f"Missing colors: {', '.join(missing_colors)}"

        expected_pattern = query.get("pattern")
        if expected_pattern and product.get("pattern") and expected_pattern not in product.get("pattern"):
            return f"Pattern mismatch: expected {expected_pattern}, got {product.get('pattern')}"

        item_type = query.get("item_type")
        title = (product.get("title") or "").lower()
        if item_type and item_type not in title:
            return f"Title does not mention {item_type}"

        return ""

    def _score_product(self, position: int, source: str) -> float:
        base = 0.9 if source == "qdrant" else 0.8
        penalty = min(position * 0.02, 0.2)
        return round(max(0.5, base - penalty), 2)
