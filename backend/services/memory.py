"""
Memory service for storing and retrieving user preferences using Mem0.
Stores: name, gender, style preferences.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from mem0 import Memory

logger = logging.getLogger(__name__)


class MemoryService:
    """Manages long-term user preferences with Mem0"""
    
    def __init__(self):
        self.memory = self._build_mem0()
    
    def _build_mem0(self) -> Memory:
        """Initialize Mem0 with Qdrant backend"""
        api_key = os.getenv("OPENAI_API_KEY")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_KEY")
        mem_collection = os.getenv("MEM_COLLECTION", "mem0_fashion_qdrant")
        
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": mem_collection,
                    "url": qdrant_url,
                    "api_key": qdrant_api_key,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "api_key": api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": api_key,
                },
            },
        }
        
        return Memory.from_config(config)
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user profile from memory.
        Returns dict with: name, gender, preferences
        """
        try:
            memories = self.memory.get_all(user_id=user_id)
            logger.info(f"[MEMORY] Retrieved {len(memories.get('results', []))} memories for user {user_id}")
            
            profile = {
                "name": self._extract_name(memories),
                "gender": self._extract_gender(memories),
                "preferences": self._extract_preferences(memories),
            }
            
            return profile
        except Exception as e:
            logger.error(f"[MEMORY] Failed to get profile for {user_id}: {e}")
            return {"name": None, "gender": None, "preferences": []}
    
    def store_user_info(self, user_id: str, message: str) -> None:
        """Store user information from conversation"""
        try:
            self.memory.add(message, user_id=user_id)
            logger.info(f"[MEMORY] Stored info for user {user_id}: '{message[:50]}'")
        except Exception as e:
            logger.error(f"[MEMORY] Failed to store info: {e}")
    
    def get_relevant_context(self, user_id: str, query: str, limit: int = 3) -> str:
        """Get relevant memories for current query"""
        try:
            results = self.memory.search(query, user_id=user_id, limit=limit)
            memories = [r.get('memory', '') for r in results.get('results', [])]
            context = " | ".join(memories)
            logger.info(f"[MEMORY] Found {len(memories)} relevant memories for query: '{query[:50]}'")
            return context
        except Exception as e:
            logger.error(f"[MEMORY] Failed to search memories: {e}")
            return ""
    
    def _extract_name(self, memories: Dict) -> Optional[str]:
        """Extract user name from memories"""
        results = memories.get('results', [])
        for mem in results:
            text = mem.get('memory', '').lower()
            # Look for patterns like "my name is X" or "I'm X" or "call me X"
            for pattern in ["my name is ", "i'm ", "i am ", "call me "]:
                if pattern in text:
                    # Extract the name (next word after pattern)
                    parts = text.split(pattern, 1)
                    if len(parts) > 1:
                        name = parts[1].split()[0].strip('.,!?').title()
                        if name and len(name) > 1:
                            return name
        return None
    
    def _extract_gender(self, memories: Dict) -> Optional[str]:
        """Extract gender preference from memories"""
        results = memories.get('results', [])
        for mem in results:
            text = mem.get('memory', '').lower()
            if any(word in text for word in ["women", "woman", "female", "ladies", "girls"]):
                return "women"
            if any(word in text for word in ["men", "male", "guys", "gentleman"]):
                return "men"
        return None
    
    def _extract_preferences(self, memories: Dict) -> List[str]:
        """Extract style preferences"""
        results = memories.get('results', [])
        preferences = []
        
        style_keywords = ["minimal", "classic", "trendy", "bold", "casual", "formal", "vintage", "modern"]
        
        for mem in results:
            text = mem.get('memory', '').lower()
            for style in style_keywords:
                if style in text and style not in preferences:
                    preferences.append(style)
        
        return preferences
