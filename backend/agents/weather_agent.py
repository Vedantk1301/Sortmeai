"""
WeatherAgent fetches weather context via Open-Meteo.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_WEATHER_CODE_MAP = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with rime",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "heavy thunderstorm with hail",
}


class WeatherAgent:
    def __init__(self, ledger_hook=None, cache_ttl: int = 600) -> None:
        self.ledger_hook = ledger_hook
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}

    def __call__(self, query_text: str, destination: str | None = None) -> Dict[str, Any]:
        # Prefer destination if available, else query_text
        location = destination or query_text
        if not location:
            return {"error": "no_location"}

        cache_key = location.lower().strip()
        now = time.time()
        if cache_key in self._cache:
            ts, val = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                return val

        try:
            result = self._fetch_open_meteo(location)
            self._cache[cache_key] = (now, result)
            
            if self.ledger_hook:
                self.ledger_hook(
                    {"location": location, "result": result}, 
                    component="weather_agent"
                )
            return result
        except Exception as e:
            logger.error(f"Weather fetch failed for {location}: {e}")
            return {"error": str(e)}

    def _fetch_open_meteo(self, location: str) -> Dict[str, Any]:
        # 1. Geocoding
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        
        if not geo_data.get("results"):
            return {"error": "location_not_found", "location": location}

        place = geo_data["results"][0]
        lat = place["latitude"]
        lon = place["longitude"]
        name = place.get("name", location)
        country = place.get("country", "")

        # 2. Weather
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code",
                "timezone": "auto",
            },
            timeout=5,
        )
        weather_resp.raise_for_status()
        w_data = weather_resp.json()
        
        current = w_data.get("current", {})
        temp = current.get("temperature_2m")
        code = current.get("weather_code")
        desc = _WEATHER_CODE_MAP.get(code, "unknown")
        
        summary = f"Currently {temp}°C and {desc} in {name}."
        if temp is not None:
            if temp > 30:
                summary += " It's hot, so wear breathable fabrics like cotton or linen."
            elif temp < 15:
                summary += " It's chilly, so layering is key."
            elif 15 <= temp <= 30:
                summary += " The weather is pleasant."

        return {
            "location": f"{name}, {country}",
            "temperature": f"{temp}°C",
            "condition": desc,
            "summary": summary,
            "raw": current
        }
