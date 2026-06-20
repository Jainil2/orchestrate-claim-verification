# Architecture

Production-grade structure, deliberately right-sized for the hackathon (judged on
quality + reproducibility, not infra weight). Core principle unchanged from the
POC: **the model perceives, deterministic code decides.**

## Data flow

```
claims.csv ─► repository.Dataset ─► per claim (bounded concurrency):
   resilience.ResultStore  ── cache/checkpoint hit? ─► reuse (idempotent resume)
        │ miss
        ▼
   vision.analyze_claim ── LLMProvider.generate_structured ──► perception JSON
        │   (few-shot + evidence-rule retrieval injected; temp=0)
        ▼
   engine.decide ── deterministic policy ──► 10 predicted fields
        │   confidence-gating · strict/lenient match · risk fusion · severity
        ▼
   (if contradicted & verifier_enabled) vision.verify_claim ──► adversarial verdict
        ▼   engine.decide(..., verdict) ── arbiter: disagreement → manual_review
   schema.validate_row ─► output.csv (14 cols, exact order)
```

Resilience wraps every model call: `ResilientProvider` = rate-limit + retry/backoff.

## Modules & responsibilities (SRP)

| Module | Responsibility | Pattern |
|---|---|---|
| `config.py` | One typed `Settings` (provider/model/keys/flags/pricing/paths) | Typed config (pydantic-settings) |
| `repository.py` | Dataset access (claims/history/evidence/images) | Repository |
| `providers/` | `LLMProvider` interface + Gemini/OpenAI-compat/Ollama adapters + factory | Strategy · Adapter · Factory |
| `vision.py` | Build perception prompt/parts; perception + verifier calls | — (provider-agnostic) |
| `engine.py` | **Pure** decision policy (no I/O, no model) | — (fully unit-tested) |
| `resilience.py` | Rate limiter, retry, result cache/checkpoint | Decorator |
| `pipeline.py` | Per-claim orchestration + failure isolation | Facade |
| `main.py` | Concurrent, resumable driver + CLI + logging | — |
| `schema.py` | 14-col contract, enums, coercion, validation | Single source of truth |
| `evaluation/` | A vs B + ablations + metrics + report | — |

## SOLID highlights
- **DIP/OCP** — everything depends on `LLMProvider`, not a vendor SDK. Add a provider = add one adapter + one factory branch; no other file changes. `providers/gemini.py` is the only file importing `google.genai`.
- **SRP** — perception (model), policy (`engine`, pure), orchestration (`pipeline`), I/O (`repository`/`main`) are separate.
- **Open/Closed** — new provider/behavior via config + new class, not edits to call sites.

## Scalability / availability
- Bounded concurrency (`ThreadPoolExecutor`) + RPM limiter.
- Retry/backoff on transient failures; per-claim failure isolation (one bad claim → safe `manual_review` row, never a crashed run).
- Result store = idempotent cache **and** checkpoint → crash-safe resume, no double spend.
- Stateless `engine.decide` (pure) → trivially parallel / horizontally scalable.

## Reproducibility
- `temperature=0`; all policy deterministic in `engine.py`.
- Offline `test_engine.py` (no key) runs in CI on every push.
- Pinned deps; provider/model/prompt-version recorded in config + report.
