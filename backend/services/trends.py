"""
Fashion trends service using Tavily and LLM summarization.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from config import Config
from services.llm import LLM

logger = logging.getLogger(__name__)

TREND_CACHE_FILE = Path("cache/fashion_trends.json")


async def get_fashion_trends_text() -> str:
    """
    Returns a compact trend summary string.
    Cached on disk for 24h.
    """
    # 1. Try disk cache
    try:
        if TREND_CACHE_FILE.exists():
            with open(TREND_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts_str = data.get("timestamp")
            text = data.get("text")
            if ts_str and text:
                ts = datetime.fromisoformat(ts_str)
                if datetime.utcnow() - ts < timedelta(seconds=Config.TRENDS_CACHE_TTL):
                    logger.info("Trend cache HIT")
                    return text
    except Exception as e:
        logger.warning(f"Trend cache read failed: {e}")

    # 2. Fetch fresh
    text = await _fetch_and_summarize_trends()
    
    # 3. Save to cache
    try:
        TREND_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TREND_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "text": text,
                },
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning(f"Trend cache write failed: {e}")

    return text


async def _fetch_and_summarize_trends() -> str:
    if not Config.TAVILY_API_KEY:
        logger.warning("Tavily API key missing, using fallback trends")
        return _fallback_trends()

    logger.info("Fetching fresh fashion trends via Tavily")
    
    try:
        western = await _tavily_search("current fashion trends western casual and streetwear India 2025")
        ethnic = await _tavily_search("current fashion trends ethnic and traditional wear India 2025")
        
        if not western and not ethnic:
            return _fallback_trends()

        summary = await _summarize_with_llm(western, ethnic)
        return summary
    except Exception as e:
        logger.error(f"Trend fetch failed: {e}")
        return _fallback_trends()


async def _tavily_search(query: str) -> List[Dict[str, Any]]:
    def _do():
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {Config.TAVILY_API_KEY}"},
            json={
                "query": query,
                "topic": "news",
                "max_results": 4,
                "include_answer": False,
                "include_images": False,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    return await asyncio.to_thread(_do)


async def _summarize_with_llm(western: List[Dict], ethnic: List[Dict]) -> str:
    llm = LLM()
    
    prompt = """You are a fashion trend summariser for an India-first stylist bot.
    
    Summarize these web results into a SHORT, structured update.
    Max 100 words total.
    
    Format:
    Western / global casual:
    - point 1
    - point 2
    
    Indian / festive:
    - point 1
    - point 2
    
    End with: "Use this only for light trend flavour."
    """
    
    context = {
        "western_results": western,
        "ethnic_results": ethnic
    }
    
    try:
        response = llm.chat(
            model=Config.FAST_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(context)}
            ],
            max_output_tokens=300
        )
        return response.strip()
    except Exception as e:
        logger.error(f"Trend summarization failed: {e}")
        return _fallback_trends()


def _fallback_trends() -> str:
    return (
        "Western / global casual:\n"
        "- Relaxed oversized tees and shirts with clean Korean style trousers\n"
        "- Straight and wide leg pants with minimal sneakers\n"
        "- Co ord sets, muted earthy tones and soft knits\n\n"
        "Indian / festive:\n"
        "- Pastel and earthy kurtas with subtle embroidery\n"
        "- Lightweight Nehru jackets and kurta co ord sets\n"
        "- Simple sherwanis and juttis with minimal details\n\n"
        "Use this only for light trend flavour."
    )
