"""
Stylist agent drafts the final user-facing response with a warm, playful tone.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from config import Config
from services.llm import LLM

EMOJIS = ["✨", "👗", "👕", "👠", "👔", "🎨", "💫", "🌟", "😊", "💃", "🕺"]

STYLIST_SYSTEM_PROMPT = """You are Sortme, a Gen Z fashion enthusiast and personal stylist AI. 
Your vibe is friendly, trendy, and helpful—like a knowledgeable bestie.
You use emojis naturally but not excessively.
You are concise and get straight to the point.
Never be robotic or overly formal.
Always be encouraging and excited about fashion.

Context:
- User Name: {name}
- Current Trends: {trends}
- Weather: {weather}
- Conversation History: {history}

Your goal is to generate a short, engaging response based on the user's situation.
If the user asks a follow-up question, use the conversation history to answer it.
"""

RICH_PRODUCT_PROMPT = """The user searched for: "{query}".
We have {count} products ready. User's name is {name}.

Write a concise, personalized reply with clear line breaks:

**Hey {name}!** Start with a short excited acknowledgment of their search (1 sentence).

**Did You Know?** 1 fun fact about the category/fabric/history that's relevant to their search.

**Styling Ideas:** 
- 3 bullet points, each with a label (e.g., Office, Weekend, Travel) + one pairing tip.

Keep it friendly, avoid walls of text, and use emojis sparingly (1–2 total).
End with an offer to refine further!
"""


BROAD_PRODUCT_PROMPT = """The user asked for: "{query}".
Context: Occasion={occasion}, Destination={destination}.
We found {count} items covering different aspects of this request.

Write a warm, helpful response in this format:

**Hey {name}!** Start with a personalized excited acknowledgment of their request (1 short sentence).

**What I Found:** Briefly describe the variety of pieces I've pulled together (e.g. "I've curated a mix of breathable tops, relaxed bottoms, and versatile pieces...") - keep it specific to what they asked for.

**Quick Styling Tips:**
- One tip about mixing these pieces
- One tip about fabric/comfort for the occasion
- One tip about accessorizing (optional)

End with a warm line like "Let me know if you want to refine by color, budget, or style! ✨"

Keep it conversational, excited, and helpful. Use 1-2 emojis max. Don't be generic!
"""

class StylistAgent:
    def __init__(self, ledger_hook=None) -> None:
        self.ledger_hook = ledger_hook
        self.llm = LLM()

    def __call__(
        self,
        products: List[Dict[str, Any]],
        query: Dict[str, Any],
        outfits: List[Dict[str, Any]] | None = None,
        weather: Dict[str, Any] | None = None,
        trends: str | None = None,
        clarification: Dict[str, Any] | None = None,
        mode: str | None = None,
        user_profile: Dict[str, Any] | None = None,
        conversation_history: List[Dict[str, str]] | None = None,
    ) -> str:
        self.user_profile = user_profile or {}
        self.trends = trends
        self.weather = weather
        self.conversation_history = conversation_history or []

        if mode == "greeting":
            return self._greeting_response()
        if mode == "capabilities_overview":
            return self._capabilities_response()
        if mode == "trending":
            return self._trending_response()
        if mode == "user_info_stored":
            return self._user_info_acknowledgment()
        if mode == "nudge":
            return self._nudge_response(query)

        if clarification and clarification.get("options") and not clarification.get("choice"):
            response = self._clarification_response(clarification)
        elif outfits:
            response = self._outfit_response(outfits, products, query, weather)
        else:
            response = self._product_response(products, query)

        if self.ledger_hook:
            self.ledger_hook({"stylist_response": response}, component="stylist")
        return response

    def _generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Helper to generate response using LLM"""
        try:
            name = self.user_profile.get("name", "there")
            trends_snippet = (self.trends or "")[:200]
            weather_snippet = (self.weather or {}).get("summary", "")
            
            # Format history for context
            history_str = ""
            for msg in self.conversation_history[-3:]: # Last 3 turns
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_str += f"{role}: {content}\n"

            system_prompt = STYLIST_SYSTEM_PROMPT.format(
                name=name,
                trends=trends_snippet,
                weather=weather_snippet,
                history=history_str
            )
            
            response = self.llm.chat(
                model=Config.FAST_MODEL, # gpt-4.1-nano
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_output_tokens=350, # Increased for rich responses
            )
            return response.strip().replace('"', '')
        except Exception as e:
            # Fallback if LLM fails
            return f"✨ Hey {name}! I'm having a bit of trouble thinking right now, but I'm still here to help you shop!"

    def _maybe_emoji(self) -> str:
        return random.choice(EMOJIS) if random.random() < 0.7 else ""

    def _product_response(self, products: List[Dict[str, Any]], query: Dict[str, Any]) -> str:
        # Generate rich text response
        raw_query = query.get("raw_query", "items")
        count = len(products)
        name = self.user_profile.get("name", "there")
        
        if query.get("query_type") == "broad":
            prompt = BROAD_PRODUCT_PROMPT.format(
                query=raw_query,
                count=count,
                name=name,
                occasion=query.get("occasion", "your event"),
                destination=query.get("destination", "your trip")
            )
            intro_text = self._generate_response(prompt, temperature=0.75)
        else:
            prompt = RICH_PRODUCT_PROMPT.format(query=raw_query, count=count, name=name)
            intro_text = self._generate_response(prompt, temperature=0.75)
        
        # Only return the rich text, let UI handle the product cards
        return intro_text

    def _nudge_response(self, query: Dict[str, Any] | None = None) -> str:
        """When user gives a low-information reply, suggest concrete starting points."""
        recent = (query or {}).get("context_hints", {}) if isinstance(query, dict) else {}
        
        prompt = "The user sent a short or unclear message. Politely nudge them to start shopping. Suggest 3 specific options like 'work-ready fits', 'casual weekend looks', or 'party outfits'. Keep it short and punchy."
        
        if recent.get("recent_occasion"):
            focus = recent["recent_occasion"]
            prompt += f" Mention their recent interest in {focus}."
            
        return self._generate_response(prompt, temperature=0.7)

    def _outfit_response(
        self,
        outfits: List[Dict[str, Any]],
        products: List[Dict[str, Any]],
        query: Dict[str, Any],
        weather: Dict[str, Any] | None,
    ) -> str:
        destination = query.get("destination") or query.get("occasion") or "your plan"
        
        # Use LLM for the intro line of outfits
        prompt = f"I just built {len(outfits)} outfits for {destination}. Write a short, excited intro sentence for these looks."
        intro = self._generate_response(prompt, temperature=0.7)
        
        lines = [intro]
        product_index = {p.get("id"): p for p in products}
        for outfit in outfits:
            names = []
            for raw in outfit.get("items", []):
                pid = raw.get("id") if isinstance(raw, dict) else raw
                names.append(product_index.get(pid, {}).get("title", pid))
            lines.append(f"- {outfit.get('name', 'Outfit')}: {outfit.get('description', '')} ({', '.join(names)})")
        return "\n".join(lines)

    def _format_product(self, product: Dict[str, Any]) -> str:
        price = product.get("price", {})
        price_str = ""
        if price.get("value") is not None:
            currency = price.get("currency") or ""
            price_str = f" - {currency}{price['value']}"
        return f"- {product.get('title', 'Product')} ({product.get('brand', 'unknown brand')}){price_str}"

    def _clarification_response(self, clarification: Dict[str, Any]) -> str:
        question = clarification.get("question") or "Can you clarify what you mean?"
        prompt = f"I need to clarify something with the user: '{question}'. Rephrase this in a friendly, helpful way."
        return self._generate_response(prompt, temperature=0.7)

    def _greeting_response(self) -> str:
        prompt = "Write a warm, energetic greeting to the user. Ask them what they want to shop for today. If you know their name, use it."
        return self._generate_response(prompt, temperature=0.8)
    
    def _trending_response(self) -> str:
        """Response for trending query - uses cached trends"""
        if not self.trends:
            return self._generate_response("I can't fetch trends right now. Apologize and ask what else I can help with.", temperature=0.7)
        
        prompt = f"Summarize these fashion trends for the user in a cool, exciting way:\n\n{self.trends[:500]}\n\nEnd by asking if they want to shop any of these looks."
        return self._generate_response(prompt, temperature=0.7)
    
    def _user_info_acknowledgment(self) -> str:
        """Response when user shares info"""
        prompt = "The user just shared their name or preferences. Acknowledge it warmly and ask what they'd like to find."
        return self._generate_response(prompt, temperature=0.7)

    def _capabilities_response(self) -> str:
        prompt = "Explain what you can do (search products, plan travel outfits, check trends, mix & match) in a fun, quick way."
        return self._generate_response(prompt, temperature=0.7)
