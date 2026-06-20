"""Resilience: rate limiting, retry/backoff, a provider decorator, and a
result store that doubles as cache + checkpoint (resume).

No heavy deps — a small hand-rolled retry (deterministic backoff, jitter from the
attempt index so reruns stay reproducible) keeps the dependency surface minimal.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from providers.base import LLMProvider, ProviderResult

# Fields that change the model output → part of the cache/checkpoint key.
_KEY_SETTINGS = ("provider", "prompt_version", "verifier_enabled",
                 "few_shot_enabled", "confidence_gating", "strict_issue_match",
                 "nei_recall_bias")


class RateLimiter:
    """Thread-safe minimum-spacing limiter (approx RPM). rpm<=0 disables it."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = now + self._interval


def with_retry(fn, *, retries: int, base_delay: float = 1.0, max_delay: float = 20.0):
    """Call fn(); retry transient exceptions with exponential backoff. Re-raises
    the last exception after `retries` attempts."""
    last = None
    for attempt in range(max(1, retries)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider/transport failure
            last = exc
            if attempt == retries - 1:
                break
            delay = min(base_delay * (2 ** attempt) + (attempt * 0.1), max_delay)
            time.sleep(delay)
    raise last


class ResilientProvider(LLMProvider):
    """Decorator: wraps any LLMProvider with rate-limiting + retry/backoff."""

    def __init__(self, inner: LLMProvider, *, rpm: int, retries: int):
        super().__init__(inner.model)
        self.name = inner.name
        self._inner = inner
        self._limiter = RateLimiter(rpm)
        self._retries = retries

    def generate_structured(self, system: str, schema: dict, parts: list[dict]) -> ProviderResult:
        def _call():
            self._limiter.acquire()
            return self._inner.generate_structured(system, schema, parts)
        return with_retry(_call, retries=self._retries)


class ResultStore:
    """Disk-backed store keyed by claim+config. Serves as both an idempotent
    result cache and a per-claim checkpoint so a crashed run resumes without
    recomputing (or re-paying for) completed claims. Disabled → no-op.
    """

    def __init__(self, cache_dir: Path, settings, enabled: bool = True):
        self.enabled = enabled
        self._dir = Path(cache_dir)
        self._sig = "|".join(f"{k}={getattr(settings, k)}" for k in _KEY_SETTINGS)
        self._sig += f"|model={settings.resolved_model()}"
        if self.enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def key(self, claim: dict) -> str:
        raw = (self._sig + "||" + claim.get("user_id", "") + "||"
               + claim.get("image_paths", "") + "||" + claim.get("user_claim", ""))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get(self, claim: dict) -> dict | None:
        if not self.enabled:
            return None
        p = self._dir / f"{self.key(claim)}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def put(self, claim: dict, row: dict) -> None:
        if not self.enabled:
            return
        p = self._dir / f"{self.key(claim)}.json"
        p.write_text(json.dumps(row), encoding="utf-8")
