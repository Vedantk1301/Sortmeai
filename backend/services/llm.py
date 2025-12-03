"""
LLM helpers wrapping the OpenAI client with safe defaults.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from config import Config


class LLM:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or Config.OPENAI_API_KEY
        self.client = OpenAI(api_key=key)

    def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """
        Thin wrapper over Responses API to keep a chat-like interface.
        - Uses low reasoning effort.
        - Drops temperature (not supported on these models).
        """
        response_format = kwargs.pop("response_format", None)
        text_format = None
        if isinstance(response_format, dict) and response_format.get("type") in ("json_object", "json_schema"):
            # Map old-style response_format to Responses API text.format
            text_format = response_format

        params: Dict[str, Any] = {
            "model": model,
            "input": messages,
            "reasoning": {"effort": "low"},
            "max_output_tokens": kwargs.pop("max_output_tokens", 2048),
        }
        if text_format:
            params["text"] = {"format": text_format}
        # Drop unsupported keys
        kwargs.pop("temperature", None)
        kwargs.pop("response_format", None)
        params.update(kwargs)
        resp = self.client.responses.create(**params)
        # Prefer output_text; fallback to first text content
        if getattr(resp, "output_text", None):
            return resp.output_text
        for item in resp.output or []:
            for part in getattr(item, "content", []) or []:
                if part.get("type") == "output_text" and part.get("text"):
                    return part["text"]
                if part.get("text"):
                    return part["text"]
        return ""

    def embed(self, inputs: List[str], model: str | None = None) -> List[List[float]]:
        emb_model = model or Config.EMB_MODEL_CATALOG
        resp = self.client.embeddings.create(model=emb_model, input=inputs)
        return [item.embedding for item in resp.data]
