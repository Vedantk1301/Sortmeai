"""
Disambiguation agent builds card payloads for human-in-the-loop clarification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DisambiguatorAgent:
    def __init__(self, ledger_hook=None) -> None:
        self.ledger_hook = ledger_hook

    def __call__(self, query: Dict[str, Any], tokens: List[str]) -> Dict[str, Any]:
        payload = {
            "type": "disambiguation",
            "question": self._build_question(query, tokens),
            "options": self._build_options(tokens),
        }
        if self.ledger_hook:
            self.ledger_hook({"cards": payload}, component="disambiguator")
        return payload

    def _build_question(self, query: Dict[str, Any], tokens: List[str]) -> str:
        color_phrase = ", ".join(query.get("colors", [])) or "these colors"
        return f"When you say {color_phrase}, what do you mean?"

    def _build_options(self, tokens: List[str]) -> List[Dict[str, Any]]:
        options: List[Dict[str, Any]] = []
        for token in tokens:
            colors = token.replace("-", " ").split()
            label_color = " + ".join(colors)
            options.append(
                {
                    "id": f"{token}-combo",
                    "label": f"{label_color} together",
                    "preview_url": f"https://example.com/{token}-combo.jpg",
                }
            )
            options.append(
                {
                    "id": f"{token}-either",
                    "label": f"Either {label_color}",
                    "preview_url": f"https://example.com/{token}-either.jpg",
                }
            )
        return options
