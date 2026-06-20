"""OpenAI-compatible hosted provider — Adapter.

Works with any endpoint that speaks the OpenAI Chat Completions API (OpenAI,
Together, Groq, OpenRouter, vLLM, LM Studio, …): switch by api_key + base_url.
Vision via base64 data-URI image_url. JSON via response_format=json_object with
the schema appended to the system prompt (maximally compatible; the deterministic
engine + schema.coerce_* tolerate loose output).
"""

from __future__ import annotations

import base64
import json

from .base import LLMProvider, ProviderResult


class OpenAICompatProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None, timeout: float | None = None):
        from openai import OpenAI  # lazy import — only when this provider is selected
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        super().__init__(model)

    def generate_structured(self, system: str, schema: dict, parts: list[dict]) -> ProviderResult:
        system_with_schema = (
            f"{system}\n\nReturn ONLY a single JSON object matching this JSON schema "
            f"(no prose, no markdown fences):\n{json.dumps(schema)}"
        )
        content = []
        for p in parts:
            if p["kind"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                b64 = base64.standard_b64encode(p["data"]).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{p['mime']};base64,{b64}"},
                })

        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": content},
            ],
        )
        text = resp.choices[0].message.content or ""
        try:
            analysis = json.loads(text)
        except json.JSONDecodeError:
            analysis = {}

        u = getattr(resp, "usage", None)
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        }
        return ProviderResult(analysis=analysis, usage=usage)
