from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

import pathlib
import sys

# Ensure local backend modules are importable when run as `uvicorn backend.main:app`
BACKEND_ROOT = pathlib.Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

from api.server import create_app
from config import Config
from langgraph import SortmeGraph, SortmeState

# Load environment variables from .env for local development
load_dotenv()

logger = logging.getLogger(__name__)


class TextQuery(BaseModel):
    query: str
    userId: Optional[str] = None
    threadId: Optional[str] = None


def build_client() -> OpenAI:
    api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")
    return OpenAI(api_key=api_key)


client = build_client()

# Reuse the LangGraph-powered FastAPI app (includes /api/chat)
app: FastAPI = create_app()
compat_graph: Optional[SortmeGraph] = getattr(app.state, "sortme_graph", None)


def parse_response_json(response: Any) -> Dict[str, Any]:
    try:
        raw = response.output_text
    except Exception as exc:  # pragma: no cover - OpenAI client surface
        raise HTTPException(status_code=500, detail="Unable to read model output.") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Model returned non-JSON output.") from exc


def _unique(seq: List[str], limit: int = 8) -> List[str]:
    seen: List[str] = []
    for item in seq:
        if not item:
            continue
        if item not in seen:
            seen.append(item)
        if len(seen) >= limit:
            break
    return seen


def _collect_colors(products: Optional[List[Dict[str, Any]]]) -> List[str]:
    colors: List[str] = []
    for product in products or []:
        vals = product.get("color") or []
        if isinstance(vals, str):
            vals = [vals]
        for val in vals:
            try:
                colors.append(str(val))
            except Exception:
                continue
    return _unique([c.strip() for c in colors if c], limit=10)


def _collect_titles(products: Optional[List[Dict[str, Any]]]) -> List[str]:
    titles = []
    for product in products or []:
        title = product.get("title")
        if title:
            titles.append(str(title))
    return _unique(titles, limit=8)


def _collect_outfit_labels(outfits: Optional[List[Dict[str, Any]]]) -> List[str]:
    labels = []
    for outfit in outfits or []:
        for key in ("title", "occasion", "vibe"):
            if outfit.get(key):
                labels.append(str(outfit[key]))
                break
    return _unique(labels, limit=6)


def _collect_keywords(products: Optional[List[Dict[str, Any]]]) -> List[str]:
    keywords: List[str] = []
    for product in products or []:
        tags = product.get("tags") or []
        if isinstance(tags, list):
            keywords.extend([str(tag) for tag in tags if tag])
    return _unique(keywords, limit=10)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze/text")
async def analyze_text(payload: TextQuery) -> Dict[str, Any]:
    """
    Compatibility endpoint that wraps the LangGraph pipeline and returns a
    simplified shape expected by the existing frontend cards.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required.")
    if compat_graph is None:
        raise HTTPException(status_code=503, detail="LangGraph agent is not available.")

    user_id = payload.userId or "compat-user"
    thread_id = payload.threadId or "compat-thread"

    state = SortmeState(user_id=user_id, user_message=payload.query)
    # Keep per-thread state to preserve clarification choices if the client reuses the thread_id
    state.thread_id = thread_id  # type: ignore[attr-defined]

    state = await compat_graph.run_once(state)

    data = {
        "summary": state.stylist_response,
        "styling_intent": state.stylist_response,
        "keywords": _collect_keywords(state.final_products),
        "colors": _collect_colors(state.final_products),
        "top_pieces": _collect_titles(state.final_products),
        "occasions": _collect_outfit_labels(state.outfits),
        "tone": "stylist",
    }

    agent_view = {
        "stylist_response": state.stylist_response,
        "products": state.final_products,
        "outfits": state.outfits,
        "clarification": {
            "question": state.clarification_question,
            "options": state.clarification_options,
        }
        if state.clarification_question
        else None,
        "disambiguation": {"options": state.disambiguation_cards} if state.disambiguation_cards else None,
        "ui_event": state.ui_event,
    }

    return {"data": data, "agent": agent_view}


@app.post("/api/analyze/profile")
async def analyze_profile(image: UploadFile = File(...)) -> Dict[str, Any]:
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    mime = image.content_type or "image/jpeg"
    b64 = base64.b64encode(contents).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    prompt = (
        "You are a senior fashion director extracting fit and palette insights from a portrait. "
        "Return ONLY JSON with keys: gender, skin_tone, undertone, age_range, best_palettes (array), "
        "style_vibes (array), fit_notes (array), pieces_to_prioritize (array), avoid (array), uplifts (array). "
        "Keep answers concise and use human-friendly language."
    )

    response = client.responses.create(
        model=Config.OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "Analyze style-ready traits. Respond with JSON only."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_output_tokens=600,
    )

    return {"data": parse_response_json(response), "response_id": response.id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
