"""LLM provider abstraction (Strategy + Factory + Adapter).

The rest of the system depends only on the LLMProvider interface, never on a
concrete SDK — so a user can choose any model (Gemini, any OpenAI-compatible
hosted endpoint, or local Ollama) via config alone.
"""

from .base import LLMProvider, ProviderResult, image_part, text_part
from .factory import get_provider

__all__ = ["LLMProvider", "ProviderResult", "text_part", "image_part", "get_provider"]
