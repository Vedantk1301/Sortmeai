"""
OutfitBuilderAgent composes validated products into coherent outfits.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from config import Config
from services.llm import LLM

OUTFIT_PROMPT = """You are a fashion outfit builder.
Given validated products and context (weather + destination rules), produce 2-5 outfits.
Constraints:
- Use only provided product ids.
- Keep colors and patterns harmonious.
- Respect weather summary (avoid heavy layers in heat, add layers in cold).
- Respect fashion_rules list if provided (cultural norms).
- Prefer 2-4 items per outfit.
- Output JSON with key 'outfits'."""


class OutfitBuilderAgent:
    def __init__(self, llm: LLM | None = None, ledger_hook=None) -> None:
        self.llm = llm or LLM()
        self.ledger_hook = ledger_hook

    def __call__(self, products: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        safe_products = [
            {k: v for k, v in p.items() if k in ("id", "title", "brand", "color", "pattern", "fit", "fabric")}
            for p in products
        ]
        payload = {"products": safe_products, "context": context}
        messages = [
            {"role": "system", "content": OUTFIT_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        content = self.llm.chat(
            model=Config.FAST_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        outfits = json.loads(content)
        outfits.setdefault("outfits", [])
        if self.ledger_hook:
            self.ledger_hook({"context": context, "outfits": outfits}, component="outfit_builder")
        return outfits
