"""
Centralised configuration for the LangGraph-backed backend.
Reads from environment variables with sensible defaults so the app can start
even if optional services (Qdrant, Tavily) are not configured locally.
"""

from __future__ import annotations

import os
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

# Load .env once so all modules relying on Config see environment values
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)
# Also load from working directory if present (no override)
load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


class Config:
    # OpenAI / model config
    OPENAI_API_KEY: str = _get_env("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = _get_env("OPENAI_MODEL", "gpt-5-nano")
    AGENT_MODEL: str = _get_env("AGENT_MODEL", OPENAI_MODEL)
    FAST_MODEL: str = _get_env("FAST_MODEL", OPENAI_MODEL)
    WEATHER_MODEL: str = _get_env("WEATHER_MODEL", OPENAI_MODEL)
    EMB_MODEL_CATALOG: str = _get_env("EMB_MODEL_CATALOG", "text-embedding-3-large")

    # Vector store / Qdrant
    QDRANT_URL: str = _get_env("QDRANT_URL", "http://localhost:6333")
    QDRANT_KEY: Optional[str] = _get_env("QDRANT_KEY") or _get_env("QDRANT_API_KEY")
    CATALOG_COLLECTION: str = _get_env("CATALOG_COLLECTION", "fashion_catalog")
    SEARCH_LIMIT: int = int(_get_env("SEARCH_LIMIT", "12"))
    HNSW_EF: int = int(_get_env("HNSW_EF", "128"))

    # Trends / research
    TAVILY_API_KEY: str = _get_env("TAVILY_API_KEY", "")
    TRENDS_CACHE_TTL: int = int(_get_env("TRENDS_CACHE_TTL", str(60 * 60 * 24)))  # 24h default

    DEBUG: bool = _get_env("DEBUG", "false").lower() == "true"
