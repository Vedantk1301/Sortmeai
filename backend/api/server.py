"""
Lightweight API server exposing the Sortme LangGraph pipeline.
Uses FastAPI if available; otherwise provides a noop stub.
"""

from __future__ import annotations

import sys
import pathlib
import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:  # pragma: no cover - keeps module importable without deps
    FastAPI = None  # type: ignore

# Absolute imports keep uvicorn happy when the project is run from the repo root
from langgraph import SortmeGraph, SortmeState

_state_store: Dict[str, SortmeState] = {}
_lock = threading.Lock()
CHAT_HISTORY_FILE = ROOT / "chat_history.json"


def _save_state():
    """Persist the state store to a JSON file."""
    try:
        data = {}
        with _lock:
            for thread_id, state in _state_store.items():
                # Convert SortmeState to dict. 
                # If using Pydantic v2, model_dump() is preferred, but dict() works for v1 and our fallback.
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = state.dict()
                data[thread_id] = state_dict
        
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved state for {len(data)} threads to {CHAT_HISTORY_FILE}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def _load_state():
    """Load the state store from a JSON file."""
    global _state_store
    if not CHAT_HISTORY_FILE.exists():
        return

    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        with _lock:
            for thread_id, state_dict in data.items():
                # Reconstruct SortmeState objects
                _state_store[thread_id] = SortmeState(**state_dict)
        logger.info(f"Loaded state for {len(_state_store)} threads from {CHAT_HISTORY_FILE}")
    except Exception as e:
        logger.error(f"Failed to load state: {e}")


def _get_state(thread_id: str, user_id: str, user_message: str) -> SortmeState:
    with _lock:
        if thread_id not in _state_store:
            _state_store[thread_id] = SortmeState(user_id=user_id, user_message=user_message)
        else:
            _state_store[thread_id].user_message = user_message
        return _state_store[thread_id]


def _apply_ui_events(state: SortmeState, ui_events: Optional[List[Dict[str, Any]]]) -> None:
    if not ui_events:
        return
    
    for event in ui_events:
        etype = event.get("type")
        payload = event.get("payload")
        
        if etype == "clarification_choice":
            state.clarification_choice = payload
            logger.info(f"[API] Applied clarification choice: {payload}")


def _build_response(state: SortmeState) -> Dict[str, Any]:
    return {
        "stylist_response": state.stylist_response,
        "products": state.final_products,
        "outfits": state.outfits,
        "clarification": {
            "question": state.clarification_question,
            "options": state.clarification_options,
        } if state.clarification_question else None,
        "disambiguation": {
            "options": state.disambiguation_cards
        } if state.disambiguation_cards else None,
        "ui_event": state.ui_event,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting up Sortme API...")
    _load_state()
    
    # Verify Qdrant connection (lightweight check)
    try:
        from ..retrievers.catalog_retriever import CatalogRetriever
        retriever = CatalogRetriever()
        if retriever.client:
            logger.info("Qdrant connection verified.")
    except Exception as e:
        logger.warning(f"Qdrant connection warning: {e}")

    yield
    
    # Shutdown
    logger.info("Shutting down Sortme API...")
    _save_state()


def create_app() -> FastAPI:
    app = FastAPI(title="Sortme API", version="1.0.0", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize graph once
    graph = SortmeGraph()
    # Expose graph on app.state for optional reuse (e.g., compatibility endpoints)
    app.state.sortme_graph = graph

    @app.post("/api/chat")
    async def chat_endpoint(payload: dict) -> dict:
        start_time = time.time()
        user_id = payload.get("userId", "demo-user")
        thread_id = payload.get("threadId", "demo-thread")
        user_message = payload.get("message", "")
        ui_events = payload.get("ui_events", [])
        
        logger.info("=" * 80)
        logger.info(f"[API] New request | user={user_id} | thread={thread_id}")
        logger.info(f"[API] Message: '{user_message}'")
        logger.info(f"[API] UI Events: {ui_events}")
        
        state = _get_state(thread_id, user_id, user_message)
        
        # Handle UI events
        event_start = time.time()
        _apply_ui_events(state, ui_events)
        event_time = time.time() - event_start
        
        # Run graph
        graph_start = time.time()
        state = await graph.run_once(state)
        graph_time = time.time() - graph_start
        
        # Build response
        response_start = time.time()
        response = _build_response(state)
        response_time = time.time() - response_start
        
        total_time = time.time() - start_time
        
        logger.info(f"[API] Request complete")
        logger.info(f"  - UI Events: {event_time:.3f}s")
        logger.info(f"  - Graph Execution: {graph_time:.3f}s")
        logger.info(f"  - Response Build: {response_time:.3f}s")
        logger.info(f"  - TOTAL: {total_time:.3f}s")
        logger.info(f"  - Products returned: {len(response.get('products', []))}")
        logger.info("=" * 80)
        
        # Save state after each request
        _save_state()
        
        return response
    
    return app


# Uvicorn entrypoint
app = create_app() if FastAPI else None
