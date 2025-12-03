"""
Factory for Qdrant client configured via environment variables.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from config import Config


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_KEY)
