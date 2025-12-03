"""
SortmeGraph wires the node flow into a sequential execution plan.
"""

from __future__ import annotations

import logging
from typing import Dict

from .nodes import (
    CatalogRetrieveNode,
    DisambiguateNode,
    ClarifierNode,
    KnowledgePlannerNode,
    MergeNode,
    MultiQueryRetrieveNode,
    OutfitBuilderNode,
    ParseNode,
    StylistNode,
    UINode,
    VisionValidateNode,
    WebFashionNode,
    WebRetrieveNode,
    WebVisionValidateNode,
    WeatherNode,
)
from .state import SortmeState
from services.user_profile import UserProfileService
from agents import StylistAgent
from services.trends import get_fashion_trends_text

logger = logging.getLogger(__name__)

# Global trends cache - fetched once on first request
_GLOBAL_TRENDS_CACHE: Dict[str, str] = {}


class SortmeGraph:
    def __init__(self) -> None:
        # User profile service with dedicated Qdrant collection
        try:
            self.profile_service = UserProfileService()
            logger.info("[GRAPH] UserProfileService initialized")
        except Exception as e:
            logger.warning(f"[GRAPH] UserProfileService init failed: {e}, profiles disabled")
            self.profile_service = None
        
        self.parse_node = ParseNode()
        self.disambiguate_node = DisambiguateNode()
        self.clarifier_node = ClarifierNode()
        self.knowledge_planner_node = KnowledgePlannerNode()
        self.weather_node = WeatherNode()
        self.web_fashion_node = WebFashionNode()
        self.multi_query_retrieve_node = MultiQueryRetrieveNode()
        self.outfit_builder_node = OutfitBuilderNode()
        self.catalog_retrieve_node = CatalogRetrieveNode()
        self.vision_validate_node = VisionValidateNode()
        self.web_retrieve_node = WebRetrieveNode()
        self.web_vision_validate_node = WebVisionValidateNode()
        self.merge_node = MergeNode()
        self.stylist_node = StylistNode()
        self.ui_node = UINode()
        self.stylist = StylistAgent() # Added

    async def run_once(self, state: SortmeState) -> SortmeState:
        # Load user profile from dedicated Qdrant collection
        if self.profile_service:
            try:
                profile = self.profile_service.get_profile(state.user_id)
                state.user_profile = profile
                name = profile.get("name")
                gender = profile.get("gender")
                logger.info(f"[GRAPH] Loaded profile: name={name}, gender={gender}")
            except Exception as e:
                logger.error(f"[GRAPH] Failed to load profile: {e}")
        
        # Load trends ONCE globally
        global _GLOBAL_TRENDS_CACHE
        if "trends" not in _GLOBAL_TRENDS_CACHE:
            logger.info("[GRAPH] Fetching trends for the first time...")
            try:
                _GLOBAL_TRENDS_CACHE["trends"] = await get_fashion_trends_text()
                logger.info("[GRAPH] Trends cached successfully")
            except Exception as e:
                logger.error(f"[GRAPH] Failed to load trends: {e}")
                _GLOBAL_TRENDS_CACHE["trends"] = ""
        
        state.trends_context = _GLOBAL_TRENDS_CACHE.get("trends", "")

        # IMPORTANT: If conversation history is empty, this is the FIRST interaction
        # Automatically show a greeting to make Sortme more inviting
        if len(state.conversation_history) == 0 and not state.user_message.strip():
            logger.info("[GRAPH] First interaction detected - showing automatic greeting")
            state.mode = "greeting"
            state = self.stylist_node(state)
            state = self.ui_node(state)
            return state

        # Add to conversation history
        state.conversation_history.append({"role": "user", "content": state.user_message})
        
        # Parse intent
        state = self.parse_node(state)
        fq = state.fashion_query or {}
        qtype = fq.get("query_type")
        intent = fq.get("intent")

        # Blocked/out-of-scope responses
        if intent == "blocked":
            responses = [
                "I'm here to help with fashion! What are you looking for?",
                "Let's keep it fashion-focused. Need help finding something?",
                "I'm your fashion assistant - what can I help you shop for?",
            ]
            import random
            state.stylist_response = random.choice(responses)
            state = self.ui_node(state)
            return state
            
        if intent == "out_of_scope":
            responses = [
                "I'm all about fashion. Looking for clothes, shoes, or complete outfits? Just let me know!",
                "That's not my specialty, but I'm great at helping you find fashion pieces. What's your style?",
                "I specialize in fashion and style. Want to browse dresses, shirts, accessories? Tell me what you need!",
                "My expertise is fashion - casual wear, party outfits, or work attire. What are you shopping for?",
            ]
            import random
            state.stylist_response = random.choice(responses)
            state = self.ui_node(state)
            return state

        # Acknowledgment (like "okay", "thanks")
        if intent == "acknowledgment":
            responses = [
                "Anything else you'd like to see?",
                "Need help finding anything else?",
                "Let me know if you want to explore more options!",
                "Want to check out something different?",
            ]
            import random
            state.stylist_response = random.choice(responses)
            state = self.ui_node(state)
            return state
        
        # User sharing info (name, gender, preferences)
        if intent == "user_info":
            import random
            if self.profile_service:
                try:
                    # Extract and save profile info
                    profile = self.profile_service.extract_and_save_from_message(state.user_id, state.user_message)
                    # Reload to get updated profile
                    state.user_profile = profile
                except Exception as e:
                    logger.error(f"[GRAPH] Failed to store user info: {e}")
            
            # Generate personalized acknowledgment with smart follow-ups
            name = (state.user_profile or {}).get("name")
            gender = (state.user_profile or {}).get("gender")
            
            response_parts = []
            
            # If they just shared their name
            if name and not gender:
                greetings = [
                    f"Nice to meet you, {name}! Are you looking for men's or women's fashion today?",
                    f"Hey {name}! Great to have you here. Should I show you men's or women's collections?",
                    f"Welcome, {name}! Quick question - are you browsing for menswear or womenswear?",
                ]
                state.stylist_response = random.choice(greetings)
                state = self.ui_node(state)
                return state
            
            # If they just shared gender (with or without name)
            elif gender:
                if name:
                    responses = [
                        f"Perfect, {name}! I'll show you great {gender}'s options. What are you looking for today?",
                        f"Got it, {name}! Ready to find some amazing {gender}'s pieces. What's the occasion?",
                        f"Awesome, {name}! Let's find you some {gender}'s fashion. What catches your eye?",
                    ]
                else:
                    responses = [
                        f"Great! I'll show you {gender}'s collections. What can I help you find?",
                        f"Perfect! Looking for something specific in {gender}'s fashion?",
                        f"Got it! What {gender}'s items are you interested in?",
                    ]
                state.stylist_response = random.choice(responses)
                state = self.ui_node(state)
                return state
            
            # Fallback
            else:
                response_parts.append("Thanks for sharing! That helps me find better matches for you.")
                response_parts.append("What can I help you find?")
                state.stylist_response = " ".join(response_parts)
                state = self.ui_node(state)
                return state

        # Greeting branch
        if qtype == "chitchat" and intent == "greeting":
            state.mode = "greeting"
            state = self.stylist_node(state)
            state = self.ui_node(state)
            return state

        # Capabilities overview
        if qtype == "capabilities":
            state.mode = "capabilities_overview"
            state = self.stylist_node(state)
            state = self.ui_node(state)
            return state

        # Broad intent path
        if qtype == "broad":
            
            state = self.knowledge_planner_node(state)
            
            # Only run weather + web search if there's a destination mentioned
            destination = (state.fashion_query or {}).get("destination")
            if destination:
                logger.info(f"[GRAPH] Destination detected: '{destination}' - running weather + web search")
                state = self.weather_node(state)
                state = self.web_fashion_node(state)
            else:
                logger.info("[GRAPH] No destination - skipping weather and web search")
            
            # Always use multi-query for broad intents (generates multiple Qdrant queries)
            state = self.multi_query_retrieve_node(state)
            state = self.outfit_builder_node(state)
            state = self.stylist_node(state)
            state = self.ui_node(state)
            return state

        # Specific fashion query
        if fq:
            state = self.disambiguate_node(state)
            if state.ui_event and state.ui_event.get("type") == "disambiguation" and not state.chosen_disambiguation:
                return state

            needs_clar = fq.get("needs_clarification", False)
            if needs_clar:
                state = self.clarifier_node(state)
                if state.clarification_options and not state.clarification_choice:
                    state = self.stylist_node(state)
                    state = self.ui_node(state)
                    return state

            state = self.catalog_retrieve_node(state)
            state = self.vision_validate_node(state)

            if len(state.qdrant_valid) < 5:
                logger.info(f"[GRAPH] Low valid products ({len(state.qdrant_valid)} < 5) -> Triggering Web Search")
                state = self.web_retrieve_node(state)
                state = self.web_vision_validate_node(state)
            else:
                logger.info(f"[GRAPH] Sufficient valid products ({len(state.qdrant_valid)} >= 5) -> Skipping Web Search")

            state = self.merge_node(state)
            state = self.stylist_node(state)
            state = self.ui_node(state)
            
            # Add bot response to conversation history
            if state.stylist_response:
                state.conversation_history.append({"role": "assistant", "content": state.stylist_response})
            
            # Keep only last 10 messages (5 turns) for context
            if len(state.conversation_history) > 10:
                state.conversation_history = state.conversation_history[-10:]
            
            return state

        # Fallback response
        state.stylist_response = "Tell me what you're looking for in fashion and I'll find options for you."
        
        # Add to conversation history
        if state.stylist_response:
            state.conversation_history.append({"role": "assistant", "content": state.stylist_response})
        
        return state
