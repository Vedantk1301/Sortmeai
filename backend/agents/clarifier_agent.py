"""
ClarifierAgent asks for user clarification on ambiguous queries (e.g., color combos).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from config import Config
from services.llm import LLM

CLARIFIER_PROMPT = """You help clarify ambiguous fashion queries.
When colour combos or similar ambiguities exist, propose 2-3 interpretations as cards.
Be conservative: if not ambiguous, set needs_clarification=false.
Output JSON: { "needs_clarification": bool, "question": str, "options": [ { "id":..., "label":..., "short_description":..., "example_constraints": { ... } } ] }
"""


class ClarifierAgent:
    def __init__(self, llm: LLM | None = None, ledger_hook=None) -> None:
        self.llm = llm or LLM()
        self.ledger_hook = ledger_hook

    def __call__(self, user_query: str, parsed_intent: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = {"query": user_query, "intent": parsed_intent or {}}
        messages = [
            {"role": "system", "content": CLARIFIER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        content = self.llm.chat(
            model=Config.AGENT_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        result = json.loads(content)
        result.setdefault("needs_clarification", False)
        result.setdefault("options", [])
        if self.ledger_hook:
            self.ledger_hook({"clarifier_result": result}, component="clarifier")
        return result
