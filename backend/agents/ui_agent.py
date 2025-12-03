"""
UIAgent generates frontend-friendly payloads such as refinement chips.
"""

from __future__ import annotations

from typing import Any, Dict, List


class UIAgent:
    def __init__(self, ledger_hook=None) -> None:
        self.ledger_hook = ledger_hook

    def refinement_cards(self, products: List[Dict[str, Any]], query: Dict[str, Any]) -> Dict[str, Any]:
        chips = [
            {"id": "more-colors", "label": "More colors"},
            {"id": "under-budget", "label": "Under Rs 13000"},
            {"id": "similar-patterns", "label": "Similar patterns"},
            {"id": "styling-tips", "label": "What to wear with this"},
        ]
        payload = {"type": "refinements", "chips": chips, "context": {"products": [p.get("id") for p in products]}}
        if self.ledger_hook:
            self.ledger_hook({"ui_event": payload}, component="ui")
        return payload

    def outfit_refinements(self, outfits: List[Dict[str, Any]], weather: Dict[str, Any] | None = None) -> Dict[str, Any]:
        chips = [
            {"id": "more-beach", "label": "More beach outfits"},
            {"id": "add-accessories", "label": "Add accessories"},
            {"id": "swap-top", "label": "Swap top"},
            {"id": "change-colors", "label": "Change colors"},
        ]
        payload = {
            "type": "outfit_refinements",
            "chips": chips,
            "context": {"outfits": [o.get("id") for o in outfits], "weather": weather},
        }
        if self.ledger_hook:
            self.ledger_hook({"ui_event": payload}, component="ui")
        return payload

    def clarification_cards(self, question: str, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "type": "clarification_cards",
            "question": question,
            "options": [
                {
                    "id": opt.get("id"),
                    "label": opt.get("label"),
                    "short_description": opt.get("short_description"),
                    "image_hint": opt.get("image_hint"),
                }
                for opt in options
            ],
        }
        if self.ledger_hook:
            self.ledger_hook({"ui_event": payload}, component="ui")
        return payload

    def capability_chips(self) -> Dict[str, Any]:
        chips = [
            {"id": "show-examples", "label": "Show example asks"},
            {"id": "plan-outfit", "label": "Plan an outfit"},
            {"id": "find-products", "label": "Find specific items"},
        ]
        payload = {"type": "capability_chips", "chips": chips}
        if self.ledger_hook:
            self.ledger_hook({"ui_event": payload}, component="ui")
        return payload
