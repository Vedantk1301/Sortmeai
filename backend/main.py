import base64
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables from .env for local development
load_dotenv()


class TextQuery(BaseModel):
    query: str


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")
    return OpenAI(api_key=api_key)


client = build_client()

app = FastAPI(
    title="Sortme AI Backend",
    description="Fashion discovery backend powered by the OpenAI Responses API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_response_json(response: Any) -> Dict[str, Any]:
    try:
        raw = response.output_text
    except Exception as exc:  # pragma: no cover - OpenAI client surface
        raise HTTPException(status_code=500, detail="Unable to read model output.") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Model returned non-JSON output.") from exc


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze/text")
async def analyze_text(payload: TextQuery) -> Dict[str, Any]:
    prompt = (
        "You are Sortme AI, a fashion discovery assistant. "
        "Analyze the user's request, then respond ONLY with a compact JSON object containing: "
        "summary, styling_intent, keywords (array), colors (array), top_pieces (array), occasions (array), tone. "
        "Stay brief and avoid brand names unless implied."
        f" User query: {payload.query}"
    )

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "Return structured JSON for fashion styling insights."}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        response_format={"type": "json_object"},
        max_output_tokens=500,
    )

    return {"data": parse_response_json(response), "response_id": response.id}


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
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
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
