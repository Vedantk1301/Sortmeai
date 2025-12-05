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
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - keeps module importable without deps
    FastAPI = None  # type: ignore
    BaseModel = None  # type: ignore

# Absolute imports keep uvicorn happy when the project is run from the repo root
from langgraph import SortmeGraph, SortmeState


# Profile update schema
class ProfileUpdate(BaseModel):
    userId: Optional[str] = None
    name: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    skin_tone: Optional[str] = None


AGE_GROUP_BUCKETS = ["16-18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def _coerce_age_group(value: Any) -> Optional[str]:
    """Normalize age group values to standard buckets."""
    if not value:
        return None
    import re
    raw = str(value).strip().lower()
    for bucket in AGE_GROUP_BUCKETS:
        if raw == bucket.lower():
            return bucket
    digits = [int(d) for d in re.findall(r"\d{1,3}", raw)]
    num = digits[0] if digits else None
    if num is not None:
        if 16 <= num <= 18: return "16-18"
        if 18 <= num <= 24: return "18-24"
        if 25 <= num <= 34: return "25-34"
        if 35 <= num <= 44: return "35-44"
        if 45 <= num <= 54: return "45-54"
        if 55 <= num <= 64: return "55-64"
        if num >= 65: return "65+"
    if "teen" in raw: return "16-18"
    if "young" in raw or "20s" in raw: return "18-24"
    if "30" in raw: return "25-34"
    if "40" in raw: return "35-44"
    if "50" in raw: return "45-54"
    if "60" in raw: return "55-64"
    if "senior" in raw or "older" in raw: return "65+"
    return None


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
        "user_profile": state.user_profile,
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
        from retrievers.catalog_retriever import CatalogRetriever  # absolute import to avoid relative issues
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
    
    # Initialize profile service
    profile_service = None
    try:
        from services.user_profile import UserProfileService
        profile_service = UserProfileService()
        logger.info("[API] UserProfileService initialized")
    except Exception as e:
        logger.warning(f"[API] UserProfileService init failed: {e}")

    @app.get("/api/profile/{userId}")
    async def get_profile(userId: str) -> dict:
        """Get stored profile fields."""
        if not userId:
            return {"data": {}}
        if not profile_service:
            return {"data": {}}
        try:
            profile = profile_service.get_profile(userId)
            return {"data": profile}
        except Exception as exc:
            logger.error(f"[PROFILE] Get failed for {userId}: {exc}")
            return {"data": {}}

    @app.post("/api/profile/update")
    async def update_profile(payload: ProfileUpdate) -> dict:
        """Update stored profile fields."""
        logger.info(f"[PROFILE] Update request: userId={payload.userId}, gender={payload.gender}, age_group={payload.age_group}, skin_tone={payload.skin_tone}")
        
        if not payload.userId:
            return {"error": "userId is required"}
            
        if not profile_service:
            return {"error": "Profile service unavailable"}
            
        try:
            profile = profile_service.get_profile(payload.userId)
            if payload.name is not None:
                profile["name"] = payload.name
            if payload.gender is not None:
                profile["gender"] = payload.gender
            if payload.age_group is not None:
                coerced = _coerce_age_group(payload.age_group)
                profile["age_group"] = coerced or payload.age_group
            if payload.skin_tone is not None:
                profile["skin_tone"] = payload.skin_tone
                
            profile_service.save_profile(payload.userId, profile)
            logger.info(f"[PROFILE] Updated successfully: {profile}")
            return {"data": profile}
        except Exception as exc:
            logger.error(f"[PROFILE] Update failed for userId={payload.userId}: {exc}")
            return {"error": str(exc)}

    @app.post("/api/chat")
    async def chat_endpoint(payload: dict) -> Any:
        try:
            from fastapi.responses import StreamingResponse
        except ImportError:
            return {"error": "StreamingResponse not available"}

        start_time = time.time()
        user_id = payload.get("userId", "demo-user")
        thread_id = payload.get("threadId", "demo-thread")
        user_message = payload.get("message", "")
        ui_events = payload.get("ui_events", [])
        
        logger.info("=" * 80)
        logger.info(f"[API] New request | user={user_id} | thread={thread_id}")
        logger.info(f"[API] Message: '{user_message}'")
        
        state = _get_state(thread_id, user_id, user_message)
        _apply_ui_events(state, ui_events)
        
        async def event_generator():
            try:
                # Callback to yield progress events
                async def status_callback(msg: str):
                    evt = {"type": "thinking", "message": msg}
                    yield f"data: {json.dumps(evt)}\n\n"
                
                # Run graph with streaming callback
                updated_state = await graph.run_once(state, status_callback=status_callback)
                
                # Build final response
                final_response = _build_response(updated_state)
                
                # Log stats
                total_time = time.time() - start_time
                logger.info(f"[API] Request complete in {total_time:.3f}s")
                logger.info(f"[API] Products returned: {len(final_response.get('products', []))}")
                
                # Save state
                _save_state()
                
                # Yield result
                result_evt = {"type": "result", "payload": final_response}
                yield f"data: {json.dumps(result_evt)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"[API] Error in stream: {e}", exc_info=True)
                err_evt = {"type": "error", "message": str(e)}
                yield f"data: {json.dumps(err_evt)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    return app


# Uvicorn entrypoint
app = create_app() if FastAPI else None
