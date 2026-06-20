"""Factory/Registry: build the configured LLMProvider from Settings."""

from __future__ import annotations

from .base import LLMProvider


def get_provider(settings) -> LLMProvider:
    """Construct the provider named by settings.provider. Lazy imports keep
    optional SDKs (openai, requests) out of the import path unless selected.
    """
    provider = settings.provider.lower()
    model = settings.resolved_model()

    if provider == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(model=model, api_key=settings.gemini_api_key, timeout=settings.timeout)

    if provider in ("openai", "openai_compat", "hosted"):
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            model=model, api_key=settings.openai_api_key,
            base_url=settings.base_url, timeout=settings.timeout)

    if provider == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(model=model, host=settings.ollama_host, timeout=settings.timeout)

    raise ValueError(
        f"Unknown provider {settings.provider!r}. Supported: gemini, openai, ollama.")
