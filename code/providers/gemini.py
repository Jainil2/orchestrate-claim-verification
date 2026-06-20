"""Gemini provider (google-genai) — Adapter to the LLMProvider interface.

This is the ONLY file allowed to import google.genai. Moves the perception call
that previously lived inline in vision.py behind the interface.
"""

from __future__ import annotations

import json

from google import genai
from google.genai import types

from .base import LLMProvider, ProviderResult


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None, timeout: float | None = None):
        super().__init__(model)
        # SDK reads GEMINI_API_KEY/GOOGLE_API_KEY if api_key is None.
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._timeout = timeout

    def generate_structured(self, system: str, schema: dict, parts: list[dict]) -> ProviderResult:
        contents = []
        for p in parts:
            if p["kind"] == "text":
                contents.append(p["text"])
            else:
                contents.append(types.Part.from_bytes(data=p["data"], mime_type=p["mime"]))

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        resp = self._client.models.generate_content(
            model=self.model, contents=contents, config=cfg,
        )

        analysis = {}
        text = getattr(resp, "text", None)
        if text:
            try:
                analysis = json.loads(text)
            except json.JSONDecodeError:
                analysis = {}

        um = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
        }
        return ProviderResult(analysis=analysis, usage=usage)
