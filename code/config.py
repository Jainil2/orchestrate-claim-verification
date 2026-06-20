"""Typed configuration — single source of truth (replaces scattered os.environ).

One `Settings` object carries provider/model/keys, concurrency/rate/timeout,
pricing, paths, and accuracy/behavior flags. Loaded once and threaded through the
app, so nothing reaches into the environment ad hoc.

Backward compatible with the existing .env (GEMINI_API_KEY / GEMINI_MODEL).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-provider default model when MODEL/GEMINI_MODEL is unset.
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",          # any OpenAI-compatible vision model
    "ollama": "llava",                # local vision model
}

# Per-provider default price ($/1M tokens) for the operational analysis.
DEFAULT_PRICING = {
    "gemini": (0.30, 2.50),
    "openai": (0.15, 0.60),
    "ollama": (0.0, 0.0),             # local = free
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "code" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- provider selection ---
    provider: str = Field("gemini", validation_alias=AliasChoices("PROVIDER", "LLM_PROVIDER"))
    model: str = Field("", validation_alias=AliasChoices("MODEL", "GEMINI_MODEL", "LLM_MODEL"))

    # --- credentials / endpoints ---
    gemini_api_key: str | None = Field(
        None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    openai_api_key: str | None = Field(
        None, validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_API_KEY"))
    base_url: str | None = Field(
        None, validation_alias=AliasChoices("OPENAI_BASE_URL", "LLM_BASE_URL"))
    ollama_host: str = Field("http://localhost:11434", validation_alias=AliasChoices("OLLAMA_HOST"))

    # --- throughput / resilience ---
    concurrency: int = Field(4, validation_alias=AliasChoices("CONCURRENCY"))
    rpm: int = Field(60, validation_alias=AliasChoices("RPM"))
    timeout: float = Field(90.0, validation_alias=AliasChoices("TIMEOUT"))
    retries: int = Field(3, validation_alias=AliasChoices("RETRIES"))

    # --- accuracy / behavior flags ---
    prompt_version: str = Field("v1", validation_alias=AliasChoices("PROMPT_VERSION"))
    verifier_enabled: bool = Field(True, validation_alias=AliasChoices("VERIFIER_ENABLED"))
    few_shot_enabled: bool = Field(True, validation_alias=AliasChoices("FEW_SHOT_ENABLED"))
    confidence_gating: bool = Field(True, validation_alias=AliasChoices("CONFIDENCE_GATING"))
    strict_issue_match: bool = Field(False, validation_alias=AliasChoices("STRICT_ISSUE_MATCH"))
    nei_recall_bias: bool = Field(True, validation_alias=AliasChoices("NEI_RECALL_BIAS"))
    manual_review_on_unparseable: bool = Field(
        True, validation_alias=AliasChoices("MANUAL_REVIEW_ON_UNPARSEABLE"))
    strict_validation: bool = Field(False, validation_alias=AliasChoices("STRICT_VALIDATION"))

    # --- caching ---
    cache_enabled: bool = Field(True, validation_alias=AliasChoices("CACHE_ENABLED"))
    cache_dir: Path = Field(REPO_ROOT / "code" / ".cache", validation_alias=AliasChoices("CACHE_DIR"))

    # --- paths ---
    dataset_dir: Path = Field(REPO_ROOT / "dataset", validation_alias=AliasChoices("DATASET_DIR"))
    output_path: Path = Field(REPO_ROOT / "output.csv", validation_alias=AliasChoices("OUTPUT_PATH"))

    # --- pricing override (optional; else per-provider default) ---
    price_in_per_mtok: float | None = Field(None, validation_alias=AliasChoices("PRICE_IN_PER_MTOK"))
    price_out_per_mtok: float | None = Field(None, validation_alias=AliasChoices("PRICE_OUT_PER_MTOK"))

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")

    def pricing(self) -> tuple[float, float]:
        din, dout = DEFAULT_PRICING.get(self.provider, (0.0, 0.0))
        return (self.price_in_per_mtok if self.price_in_per_mtok is not None else din,
                self.price_out_per_mtok if self.price_out_per_mtok is not None else dout)


def load_settings(**overrides) -> Settings:
    """Construct Settings from env/.env, with optional explicit overrides (CLI flags)."""
    return Settings(**{k: v for k, v in overrides.items() if v is not None})
