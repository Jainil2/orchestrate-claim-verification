"""LLMProvider interface + provider-neutral content parts.

A "part" is a dict the caller builds without knowing the provider:
  text_part("hello")                     -> {"kind": "text", "text": "hello"}
  image_part(b"...", "image/jpeg")       -> {"kind": "image", "data": ..., "mime": ...}

Each concrete provider translates parts into its own SDK shape. The interface
returns a provider-neutral ProviderResult so perception/verification code and the
deterministic engine never see a vendor type.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


def text_part(text: str) -> dict:
    return {"kind": "text", "text": text}


def image_part(data: bytes, mime: str) -> dict:
    return {"kind": "image", "data": data, "mime": mime}


@dataclass
class ProviderResult:
    analysis: dict                       # parsed JSON object from the model
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


class LLMProvider(abc.ABC):
    """Strategy interface. One method: structured (JSON) generation from multimodal parts."""

    name: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    def generate_structured(self, system: str, schema: dict, parts: list[dict]) -> ProviderResult:
        """Run one call: system prompt + multimodal parts -> JSON matching `schema`.

        Implementations must return ProviderResult; on an unparseable response,
        return ProviderResult(analysis={}, ...) rather than raising for parse-only
        failures (transport/HTTP errors should still raise so the resilience layer
        can retry).
        """
        raise NotImplementedError
