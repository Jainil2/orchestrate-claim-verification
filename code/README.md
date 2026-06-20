# Multi-Modal Claim Verification

Verifies damage claims (car / laptop / package) by reading the claim conversation
and submitted images against per-object evidence requirements and user history.

**Core rule:** images are the source of truth, the conversation defines the claim,
user history adds risk context only — it never overrides clear visual evidence.

**Design:** the model *perceives* (structured JSON: object/parts/damage/quality/
authenticity); deterministic Python *decides* all policy. Reproducible + auditable.
See `ARCHITECTURE.md` for patterns and `evaluation/JUDGE_NOTES.md` for rationale.

## Provider-agnostic — choose any model

Selected by config (env/.env or `--provider`):

| `PROVIDER` | What | Needs |
|---|---|---|
| `gemini` (default) | Google Gemini (`gemini-2.5-flash`) | `GEMINI_API_KEY` |
| `openai` | any OpenAI-compatible hosted model | `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`), `pip install openai` |
| `ollama` | local vision model, offline | Ollama running + a vision model (e.g. `ollama pull llava`) |

Only `providers/gemini.py` imports a vendor SDK; everything else depends on the
`LLMProvider` interface (`providers/base.py`).

## Setup

```bash
pip install -r code/requirements.txt
cp code/.env.example .env && edit .env       # set GEMINI_API_KEY (or your provider's)
```

Secrets are read from env only via `config.Settings`. Key knobs (all overridable in
`.env`): `PROVIDER`, `MODEL`, `CONCURRENCY`, `RPM`, `RETRIES`, `VERIFIER_ENABLED`,
`FEW_SHOT_ENABLED`, `CONFIDENCE_GATING`, `STRICT_ISSUE_MATCH`, `NEI_RECALL_BIAS`,
`CACHE_ENABLED`, `STRICT_VALIDATION`.

## Run

```bash
# Offline self-test (no API key) — the deterministic engine
python code/test_engine.py

# Smoke test (needs a provider key)
python code/main.py --limit 3

# Full test set -> output.csv (44 rows), concurrent + resumable
python code/main.py

# Local model
PROVIDER=ollama python code/main.py --limit 2

# Per-claim reasoning dump (perception + verifier + engine + row)
python code/main.py --trace

# Evaluate on labeled sample (A monolithic vs B hybrid + ablations) -> evaluation_report.md
python code/evaluation/main.py --ablations
```

Re-running is cheap and idempotent: completed claims are served from the result
store (`code/.cache/`), so a crash mid-run resumes without recomputing or re-paying.

## Modules

`config.py` (typed settings) · `repository.py` (dataset access) · `providers/`
(LLMProvider + adapters + factory) · `vision.py` (perception + verifier) ·
`engine.py` (pure decision policy) · `resilience.py` (retry/cache/rate-limit) ·
`pipeline.py` (orchestration) · `main.py` (concurrent driver) · `schema.py`
(14-col contract) · `evaluation/` (scorer + A/B/ablations).

## Output schema

14 columns in exact order: `user_id, image_paths, user_claim, claim_object`
(passthrough) + `evidence_standard_met, evidence_standard_met_reason, risk_flags,
issue_type, object_part, claim_status, claim_status_justification,
supporting_image_ids, valid_image, severity`. Enforced by `schema.validate_row`;
allowed values defined in `problem_statement.md`. No labels/claim-IDs hardcoded.
