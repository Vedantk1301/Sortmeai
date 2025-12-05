from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional

import pathlib
import sys

# Ensure local backend modules are importable when run as `uvicorn backend.main:app`
BACKEND_ROOT = pathlib.Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

from api.server import create_app
from config import Config
from langgraph import SortmeGraph, SortmeState
from services.user_profile import UserProfileService

# Load environment variables from .env for local development
load_dotenv()

logger = logging.getLogger(__name__)

# Controlled age buckets for vision outputs
AGE_GROUP_BUCKETS = [
    "16-18",
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]


class TextQuery(BaseModel):
    query: str
    userId: Optional[str] = None
    threadId: Optional[str] = None


class ProfileUpdate(BaseModel):
    userId: Optional[str] = None
    name: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    skin_tone: Optional[str] = None


def build_client() -> OpenAI:
    api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")
    return OpenAI(api_key=api_key)


client = build_client()
profile_service: Optional[UserProfileService] = None

# Reuse the LangGraph-powered FastAPI app (includes /api/chat)
app: FastAPI = create_app()
compat_graph: Optional[SortmeGraph] = getattr(app.state, "sortme_graph", None)


def parse_response_json(response: Any) -> Dict[str, Any]:
    try:
        raw = response.output_text
    except Exception as exc:  # pragma: no cover - OpenAI client surface
        raise HTTPException(status_code=500, detail="Unable to read model output.") from exc

    # Best-effort JSON parsing with fallback extraction
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt to extract the first JSON object if the model added stray text
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.error(f"[VISION] Failed to parse JSON from response: {raw[:200]}...")
            raise HTTPException(status_code=500, detail="Model returned non-JSON output.") from exc

    logger.error(f"[VISION] No JSON object found in response: {raw[:200]}...")
    raise HTTPException(status_code=500, detail="Model returned non-JSON output.")


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


def _coerce_age_group(value: Any) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip().lower()
    # direct hit
    for bucket in AGE_GROUP_BUCKETS:
        if raw == bucket.lower():
            return bucket
    # number-driven bucketing
    digits = [int(d) for d in re.findall(r"\d{1,3}", raw)]
    num = digits[0] if digits else None
    if num is not None:
        if 16 <= num <= 18:
            return "16-18"
        if 18 <= num <= 24:
            return "18-24"
        if 25 <= num <= 34:
            return "25-34"
        if 35 <= num <= 44:
            return "35-44"
        if 45 <= num <= 54:
            return "45-54"
        if 55 <= num <= 64:
            return "55-64"
        if num >= 65:
            return "65+"
    # fuzzy contains for text descriptions
    if "teen" in raw:
        return "16-18"
    if "young adult" in raw or "early twenties" in raw or "twenties" in raw or "20s" in raw:
        return "18-24"
    if "30" in raw or "thirty" in raw:
        return "25-34"
    if "40" in raw or "forty" in raw:
        return "35-44"
    if "50" in raw or "fifty" in raw:
        return "45-54"
    if "60" in raw or "sixty" in raw:
        return "55-64"
    if "senior" in raw or "older" in raw or "retired" in raw:
        return "65+"
    return None


def _normalize_profile_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the vision output so the UI and future LangGraph mapping stay stable.
    Adds defaults, renames legacy keys, and marks fallback states when the photo isn't usable.
    """
    profile: Dict[str, Any] = dict(raw or {})

    # Prefer age_group, but support legacy age_range then coerce into allowed buckets
    age_value = profile.get("age_group") or profile.pop("age_range", None)
    profile["age_group"] = _coerce_age_group(age_value)

    # Flags for image quality
    is_person = bool(profile.get("is_person", True))
    quality = str(profile.get("quality", "")).lower() or None
    fallback_reason = profile.get("fallback_reason")

    # Ensure list fields are always arrays
    for key in ["best_palettes", "style_vibes", "fit_notes", "pieces_to_prioritize", "avoid", "uplifts"]:
        val = profile.get(key)
        if not isinstance(val, list):
            profile[key] = [v for v in [val] if v] if val else []

    # Mark unusable photos so the client can show a friendly fallback
    quality_flag = quality or ""
    needs_new_photo = not is_person or ("low" in quality_flag) or ("blurry" in quality_flag)
    if needs_new_photo:
        profile["status"] = "needs_new_photo"
        profile["message"] = fallback_reason or "I couldn't clearly see a person in that photo. Try a clearer face shot."
    else:
        profile["status"] = "ok"
        profile["message"] = profile.get("message") or "Profile updated from your photo."

    profile["is_person"] = is_person
    profile["quality"] = quality_flag or "unknown"
    return profile


def _persist_profile_fields(user_id: Optional[str], profile: Dict[str, Any]) -> None:
    """
    Store gender/age/skin tone for future LangGraph sessions when available.
    Never mutates the user's name.
    """
    global profile_service
    if not user_id:
        return
    if profile.get("status") == "needs_new_photo":
        return

    # Lazily init service so tests/dev without Qdrant don't crash the endpoint
    if profile_service is None:
        try:
            profile_service = UserProfileService()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[PROFILE] Could not init profile service: {exc}")
            profile_service = False  # sentinel to avoid retry storms
    if profile_service is False:
        return

    updates: Dict[str, Any] = {}
    for field in ("gender", "age_group", "skin_tone"):
        if profile.get(field):
            updates[field] = profile[field]
    if not updates:
        return

    try:
        existing = profile_service.get_profile(user_id)  # type: ignore[union-attr]
        existing.update(updates)
        profile_service.save_profile(user_id, existing)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[PROFILE] Failed to persist vision profile for {user_id}: {exc}")


def _get_profile_service() -> Optional[UserProfileService]:
    """Ensure we have a profile service instance or return None if unavailable."""
    global profile_service
    if profile_service is None:
        try:
            profile_service = UserProfileService()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[PROFILE] Could not init profile service: {exc}")
            profile_service = False
    if profile_service is False:
        return None
    return profile_service


def _should_rate_limit_upload(user_id: Optional[str]) -> Optional[str]:
    """
    Enforce a simple per-user photo analysis cap (3 per day).
    Returns an error message if the user is over the limit, otherwise None.
    """
    if not user_id:
        return None
    service = _get_profile_service()
    if not service:
        return None

    today = date.today().isoformat()
    profile = service.get_profile(user_id)
    count = int(profile.get("photo_upload_count") or 0)
    last_day = profile.get("photo_upload_date")

    limit = getattr(Config, "PHOTO_UPLOAD_LIMIT", 3)
    if last_day == today and count >= limit:
        return f"Daily photo analysis limit reached ({limit} per day). Try again tomorrow."
    return None


def _record_successful_upload(user_id: Optional[str]) -> None:
    """Increment per-day upload counter after a successful analysis."""
    if not user_id:
        return
    service = _get_profile_service()
    if not service:
        return

    today = date.today().isoformat()
    profile = service.get_profile(user_id)
    count = int(profile.get("photo_upload_count") or 0)
    last_day = profile.get("photo_upload_date")
    if last_day != today:
        count = 0
    profile["photo_upload_date"] = today
    profile["photo_upload_count"] = count + 1

    try:
        service.save_profile(user_id, profile)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[PROFILE] Failed to record upload count for {user_id}: {exc}")


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


@app.get("/api/profile/{userId}")
async def get_profile(userId: str) -> Dict[str, Any]:
    """Get stored profile fields."""
    if not userId:
        raise HTTPException(status_code=400, detail="userId is required.")

    service = _get_profile_service()
    if not service:
        # Fallback if service is unavailable (e.g. no Qdrant)
        return {"data": {}}

    try:
        profile = service.get_profile(userId)
        return {"data": profile}
    except Exception as exc:
        logger.error(f"[PROFILE] Get failed for {userId}: {exc}")
        return {"data": {}}


@app.post("/api/profile/update")
async def update_profile(payload: ProfileUpdate) -> Dict[str, Any]:
    """Update stored profile fields (gender, age_group, skin_tone, optional name)."""
    if not payload.userId:
        raise HTTPException(status_code=400, detail="userId is required.")

    service = _get_profile_service()
    if not service:
        raise HTTPException(status_code=503, detail="Profile service unavailable.")

    try:
        profile = service.get_profile(payload.userId)
        if payload.name is not None:
            profile["name"] = payload.name
        if payload.gender is not None:
            profile["gender"] = payload.gender
        if payload.age_group is not None:
            profile["age_group"] = _coerce_age_group(payload.age_group) or payload.age_group
        if payload.skin_tone is not None:
            profile["skin_tone"] = payload.skin_tone

        saved = service.save_profile(payload.userId, profile)
        if not saved:
            raise HTTPException(status_code=500, detail="Failed to save profile.")
        return {"data": profile}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[PROFILE] Update failed for {payload.userId}: {exc}")
        raise HTTPException(status_code=500, detail="Could not update profile.") from exc


@app.post("/api/analyze/profile")
async def analyze_profile(
    image: UploadFile = File(...),
    userId: Optional[str] = Form(None),
    threadId: Optional[str] = Form(None),  # reserved for future mapping if needed
) -> Dict[str, Any]:
    limit_reason = _should_rate_limit_upload(userId)
    if limit_reason:
        raise HTTPException(status_code=429, detail=limit_reason)

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    mime = image.content_type or "image/jpeg"
    b64 = base64.b64encode(contents).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    prompt = (
        "You are a senior fashion director extracting fit and palette insights from a portrait. "
        "If you cannot see a clear human, set is_person to false and explain in fallback_reason. "
        "Return ONLY JSON with keys: is_person (boolean), quality ('high'|'medium'|'low'), fallback_reason (string|null), "
        "gender, skin_tone, age_group, best_palettes (array), style_vibes (array), fit_notes (array), pieces_to_prioritize (array), avoid (array), uplifts (array). "
        f"For age_group choose exactly one closest option from this list: {', '.join(AGE_GROUP_BUCKETS)}. "
        "Keep answers concise, honest, and human-friendly."
    )

    response = client.responses.create(
        model=Config.OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Analyze style-ready traits. Respond with JSON only, no prose, no markdown.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        max_output_tokens=600,
    )

    parsed = parse_response_json(response)
    profile = _normalize_profile_payload(parsed)

    # Record successful analysis before persisting fields
    _record_successful_upload(userId)

    # Persist gender/age/skin tone for future sessions (mirrors how we keep name/gender today)
    _persist_profile_fields(userId, profile)

    return {"data": profile, "response_id": response.id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
