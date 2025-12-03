"""
Stylist agent drafts the final user-facing response with a warm, playful tone.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

EMOJIS = ["✨", "👗", "👕", "👠", "👔", "🎨", "💫", "🌟", "😊", "💃", "🕺"]


class StylistAgent:
    def __init__(self, ledger_hook=None) -> None:
        self.ledger_hook = ledger_hook

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
    ) -> str:
        self.user_profile = user_profile or {}
        self.trends = trends
        
        if mode == "greeting":
            return self._greeting_response()
        if mode == "capabilities_overview":
            return self._capabilities_response()
        if mode == "user_info_stored":
            return self._user_info_acknowledgment()

        if clarification and clarification.get("options") and not clarification.get("choice"):
            response = self._clarification_response(clarification)
        elif outfits:
            response = self._outfit_response(outfits, products, query, weather)
        else:
            response = self._product_response(products, query)

        if self.ledger_hook:
            self.ledger_hook({"stylist_response": response}, component="stylist")
        return response

    def _maybe_emoji(self) -> str:
        return random.choice(EMOJIS) if random.random() < 0.7 else ""

    def _product_response(self, products: List[Dict[str, Any]], query: Dict[str, Any]) -> str:
        intro = self._intro_line(query, products)
        body_lines = [self._format_product(prod) for prod in products]
        return "\n".join([intro, *body_lines])

    def _outfit_response(
        self,
        outfits: List[Dict[str, Any]],
        products: List[Dict[str, Any]],
        query: Dict[str, Any],
        weather: Dict[str, Any] | None,
    ) -> str:
        destination = query.get("destination") or query.get("occasion") or "your plan"
        weather_hint = ""
        if weather and weather.get("summary"):
            weather_hint = f" Weather check: {weather['summary']}."
        lines = [f"{self._maybe_emoji()} Built {len(outfits)} outfits for {destination}.{weather_hint}"]
        product_index = {p.get("id"): p for p in products}
        for outfit in outfits:
            names = [product_index.get(pid, {}).get("title", pid) for pid in outfit.get("items", [])]
            lines.append(f"- {outfit.get('name', 'Outfit')}: {outfit.get('description', '')} ({', '.join(names)})")
        return "\n".join(lines)

    def _intro_line(self, query: Dict[str, Any], products: List[Dict[str, Any]]) -> str:
        color_phrase = ", ".join(query.get("colors", [])) if query else ""
        item_type = query.get("item_type", "pieces") if query else "pieces"
        count = len(products)
        emoji = self._maybe_emoji()
        
        # More conversational intros
        intros = [
            f"{emoji} Found {count} {color_phrase} {item_type} that match what you're looking for!",
            f"{emoji} I've got {count} {color_phrase} {item_type} picks for you:",
            f"{emoji} Here are {count} {color_phrase} {item_type} I think you'll love:",
            f"{emoji} Check these out - {count} {color_phrase} {item_type} that fit the vibe:",
        ]
        intro = random.choice(intros)
        
        # Optional trend tip
        if random.random() < 0.3: # 30% chance
            tip = self._get_random_trend_tip()
            if tip:
                intro += f"\n\n{tip}"
        
        return intro

    def _get_random_trend_tip(self) -> str:
        if not self.trends:
            return ""
        import re
        # Extract lines starting with "- "
        tips = re.findall(r"-\s+(.*)", self.trends)
        if tips:
            return f"💡 Trend Tip: {random.choice(tips)}"
        return ""

    def _format_product(self, product: Dict[str, Any]) -> str:
        price = product.get("price", {})
        price_str = ""
        if price.get("value") is not None:
            currency = price.get("currency") or ""
            price_str = f" - {currency}{price['value']}"
        return f"- {product.get('title', 'Product')} ({product.get('brand', 'unknown brand')}){price_str}"

    def _clarification_response(self, clarification: Dict[str, Any]) -> str:
        question = clarification.get("question") or "Can you clarify what you mean?"
        responses = [
            f"{self._maybe_emoji()} Just want to make sure I get this right! {question}",
            f"{self._maybe_emoji()} Quick question to nail the perfect picks: {question}",
            f"{self._maybe_emoji ()} Let's narrow this down a bit. {question}",
        ]
        return random.choice(responses)

    def _greeting_response(self) -> str:
        name = self.user_profile.get("name")
        emoji = self._maybe_emoji()
        
        if name:
            greetings = [
                f"{emoji} Hey {name}! So good to see you again! What are we shopping for today?",
                f"{emoji} Welcome back, {name}! Ready to find something amazing?",
                f"{emoji} Hey {name}! I've been waiting for you! What's on your fashion wishlist today?",
            ]
            return random.choice(greetings)
        else:
            intros = [
                f"Hey! {emoji} I'm Sortme, your personal AI fashion stylist! I'm here to help you discover amazing outfits, build complete looks for any occasion, and find pieces that match your vibe. What's your name?",
                f"Hi there! {emoji} I'm Sortme! Think of me as your fashion bestie who knows every piece in our catalog. I can help you find outfits for trips, events, or just everyday slaying. Before we dive in - what should I call you?",
                f"Hey! {emoji} I'm Sortme, and I'm SO excited to help you find your perfect style! I specialize in creating complete outfits, matching colors and patterns, and finding exactly what you're looking for. What's your name?",
            ]
            return random.choice(intros)
    
    def _user_info_acknowledgment(self) -> str:
        """Response when user shares info"""
        name = self.user_profile.get("name")
        gender = self.user_profile.get("gender")
        
        responses = []
        if name:
            responses.append(f"✨ Nice to meet you, {name}!")
        if gender:
            responses.append(f"Got it - I'll show you {gender}'s collections!")
        
        if not responses:
            responses.append("✨ Thanks for sharing! That helps me find better matches for you.")
        
        responses.append("What can I help you find?")
        return " ".join(responses)

    def _capabilities_response(self) -> str:
        emoji = self._maybe_emoji()
        return (
            f"{emoji} I can help you with product searches for specific items like 'blue linen shirt' or 'black dress', "
            f"plan travel outfits for trips like 'Dubai in March' or 'Kashmir next week', "
            f"get outfit ideas for parties, work, dates, or festivals, "
            f"and mix and match colors, patterns, and complete looks. "
            f"Just tell me what you're looking for, and I'll find the perfect options!"
        )
