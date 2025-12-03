"""
Core LangGraph state definitions for the Sortme retrieval system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # Lightweight fallback to keep scaffolding importable
    class _Field:
        def __init__(self, default: Any = None, default_factory=None, description: str = "") -> None:
            self.default = default
            self.default_factory = default_factory
            self.description = description

    def Field(default: Any = None, default_factory=None, description: str = "") -> Any:  # type: ignore  # noqa: D401
        return _Field(default=default, default_factory=default_factory, description=description)

    class BaseModel:  # type: ignore
        def __init__(self, **data: Any) -> None:
            annotations = getattr(self, "__annotations__", {})
            for field_name, _ in annotations.items():
                attr = getattr(self.__class__, field_name, None)
                if isinstance(attr, _Field):
                    if field_name not in data:
                        if attr.default_factory is not None:
                            setattr(self, field_name, attr.default_factory())
                        else:
                            setattr(self, field_name, attr.default)
                        continue
                if field_name in data:
                    setattr(self, field_name, data[field_name])

            for key, value in data.items():
                if not hasattr(self, key):
                    setattr(self, key, value)

        def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # noqa: D401
            return dict(vars(self))


class LedgerEvent(BaseModel):
    component: str
    payload: Dict[str, Any]
    label: str = "llm_call"


class SortmeState(BaseModel):
    user_id: str
    user_message: str

    # Memory and conversation
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Recent conversation messages")
    user_profile: Optional[Dict[str, Any]] = Field(default=None, description="User preferences from Mem0")

    fashion_query: Optional[Dict[str, Any]] = None
    broad_intent: Optional[Dict[str, Any]] = None
    planner_plan: Optional[Dict[str, Any]] = None
    weather_context: Optional[Dict[str, Any]] = None
    trends_context: Optional[str] = None
    fashion_knowledge: Optional[Dict[str, Any]] = None
    mode: Optional[str] = None

    ambiguities: List[Dict[str, Any]] = Field(default_factory=list)
    disambiguation_cards: List[Dict[str, Any]] = Field(default_factory=list)
    chosen_disambiguation: Optional[str] = None

    clarification_options: Optional[List[Dict[str, Any]]] = None
    clarification_question: Optional[str] = None
    clarification_choice: Optional[str] = None
    clarification_source_query: Optional[str] = None
    interpretation_flags: Dict[str, Any] = Field(default_factory=dict)

    qdrant_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    qdrant_filtered: List[Dict[str, Any]] = Field(default_factory=list)
    qdrant_valid: List[Dict[str, Any]] = Field(default_factory=list)

    web_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    web_valid: List[Dict[str, Any]] = Field(default_factory=list)

    pooled_valid_products: List[Dict[str, Any]] = Field(default_factory=list)
    outfits: List[Dict[str, Any]] = Field(default_factory=list)
    final_products: List[Dict[str, Any]] = Field(default_factory=list)

    stylist_response: Optional[str] = None
    ui_event: Optional[Dict[str, Any]] = None

    ledger: List[LedgerEvent] = Field(default_factory=list)

    def log_event(self, component: str, payload: Dict[str, Any], label: str = "llm_call") -> None:
        self.ledger.append(LedgerEvent(component=component, payload=payload, label=label))

    def update_with(self, **kwargs: Any) -> "SortmeState":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
