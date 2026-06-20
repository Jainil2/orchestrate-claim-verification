"""Local Ollama provider — Adapter (degraded JSON mode).

Talks to a local Ollama server (/api/chat) using stdlib urllib (no extra dep).
Local vision models (LLaVA, Qwen-VL, …) are weaker than hosted ones and may
ignore strict schemas, so we request format="json" + put the schema in the prompt
and lean on schema.coerce_* downstream. Runs fully offline; cost = $0.
"""

from __future__ import annotations

import base64
import json
import urllib.request

from .base import LLMProvider, ProviderResult


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout: float | None = None):
        super().__init__(model)
        self._host = host.rstrip("/")
        self._timeout = timeout or 120.0

    def generate_structured(self, system: str, schema: dict, parts: list[dict]) -> ProviderResult:
        system_with_schema = (
            f"{system}\n\nReturn ONLY a single JSON object matching this JSON schema "
            f"(no prose, no markdown fences):\n{json.dumps(schema)}"
        )
        text_chunks, images_b64 = [], []
        for p in parts:
            if p["kind"] == "text":
                text_chunks.append(p["text"])
            else:
                images_b64.append(base64.standard_b64encode(p["data"]).decode("utf-8"))

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": "\n".join(text_chunks), "images": images_b64},
            ],
        }
        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as r:  # raises on HTTP error
            body = json.loads(r.read().decode("utf-8"))

        content = (body.get("message") or {}).get("content", "") or ""
        try:
            analysis = json.loads(content)
        except json.JSONDecodeError:
            analysis = {}

        usage = {
            "input_tokens": body.get("prompt_eval_count", 0) or 0,
            "output_tokens": body.get("eval_count", 0) or 0,
        }
        return ProviderResult(analysis=analysis, usage=usage)
