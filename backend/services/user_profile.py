"""
Dedicated user profile storage using a structured Qdrant collection.
Bypasses mem0 for direct, reliable profile storage.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from config import Config

logger = logging.getLogger(__name__)


class UserProfileService:
    """
    Direct Qdrant storage for user profiles.
    Each user gets a single point with their profile data.
    """
    
    COLLECTION_NAME = "muse_user_profiles"
    
    def __init__(self, client: QdrantClient | None = None):
        from services.qdrant_client import get_qdrant_client
        self.client = client or get_qdrant_client()
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.COLLECTION_NAME for c in collections)
            
            if not exists:
                logger.info(f"[PROFILE] Creating collection: {self.COLLECTION_NAME}")
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config={},  # No vectors needed, pure payload storage
                )
                
                # Create index on user_id for fast lookup
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name="user_id",
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"[PROFILE] Collection created successfully")
        except Exception as e:
            logger.error(f"[PROFILE] Failed to ensure collection: {e}")
    
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile by user_id"""
        try:
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="user_id",
                            match=rest.MatchValue(value=user_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            
            points = results[0]
            if points:
                profile = points[0].payload or {}
                logger.info(f"[PROFILE] Loaded profile for {user_id}: {profile.get('name')}")
                return profile
            
            logger.info(f"[PROFILE] No profile found for {user_id}")
            return {}
            
        except Exception as e:
            logger.error(f"[PROFILE] Failed to get profile for {user_id}: {e}")
            return {}
    
    def save_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Save or update user profile"""
        try:
            # Add metadata
            profile_data["user_id"] = user_id
            profile_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Use user_id as point ID for easy updates
            point_id = abs(hash(user_id)) % (10 ** 10)
            
            # Upsert (create or update)
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    rest.PointStruct(
                        id=point_id,
                        payload=profile_data,
                        vector={},  # Empty vector for non-vector collection
                    )
                ],
            )
            
            logger.info(f"[PROFILE] Saved profile for {user_id}: {profile_data}")
            return True
            
        except Exception as e:
            logger.error(f"[PROFILE] Failed to save profile for {user_id}: {e}")
            return False
    
    def update_field(self, user_id: str, field: str, value: Any) -> bool:
        """Update a single field in user profile"""
        profile = self.get_profile(user_id)
        profile[field] = value
        return self.save_profile(user_id, profile)
    
    def extract_and_save_from_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Extract profile info from user message and save it"""
        profile = self.get_profile(user_id)
        updated = False
        
        # Simple extraction (can be enhanced with LLM)
        lowered = message.lower()
        
        # Extract name
        if "my name is" in lowered or "i am" in lowered or "call me" in lowered:
            # Simple name extraction
            keywords = ["my name is", "i am", "call me", "i'm"]
            for kw in keywords:
                if kw in lowered:
                    parts = lowered.split(kw, 1)
                    if len(parts) > 1:
                        name_part = parts[1].strip().split()[0]
                        # Capitalize
                        name = name_part.strip(".,!?").capitalize()
                        if name and len(name) > 1:
                            profile["name"] = name
                            updated = True
                            logger.info(f"[PROFILE] Extracted name: {name}")
                            break
        
        # Extract gender preference
        gender_map = {
            "men": "men",
            "man": "men",
            "menswear": "men",
            "male": "men",
            "women": "women",
            "woman": "women",
            "womenswear": "women",
            "female": "women",
            "ladies": "women",
        }
        
        for key, gender in gender_map.items():
            if key in lowered:
                profile["gender"] = gender
                updated = True
                logger.info(f"[PROFILE] Extracted gender: {gender}")
                break
        
        if updated:
            self.save_profile(user_id, profile)
        
        return profile
