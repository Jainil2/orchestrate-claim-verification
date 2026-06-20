# Judge Notes — design rationale & defensibility

## The one idea
**The model perceives; deterministic code decides.** The LLM only reports what it
sees (objects, parts, damage, quality, authenticity) as structured JSON. Every
policy call — evidence sufficiency, supported/contradicted/insufficient, risk,
severity, supporting-image attribution — is plain, unit-tested Python (`engine.py`).
This makes outputs reproducible (temp=0 + deterministic rules) and auditable, and
it lets us guarantee the core rule: **images win; user history only adds risk
context, it never flips a visual decision.**

## Architecture & model choices
- **Provider-agnostic** (`providers/`, Strategy+Factory+Adapter): run Gemini (default),
  any OpenAI-compatible hosted model (swap by API key/base_url), or a **local Ollama**
  vision model — by config alone. Only `providers/gemini.py` touches a vendor SDK.
- **Gemini 2.5 Flash** by default: strong vision + JSON mode, cheap, fast — the
  cost/quality sweet spot at this scale. Tier is one env var away.

## Accuracy strategy (eval-driven)
1. **Confidence-gating** — the model already emits per-damage confidence; low confidence
   routes to `manual_review` / `not_enough_information` instead of a silent assertion.
2. **Few-shot + evidence-rule retrieval** injected into the perception prompt.
3. **Adversarial verifier** on the high-stakes `contradicted` path (asserting the user
   is wrong): an independent second look; on disagreement the arbiter refuses to
   contradict and routes to manual review. Cost-bounded (only contested claims).
See `evaluation_report.md` for A-vs-B + ablation deltas.

## Failure handling
Per-claim isolation (one failure → safe `manual_review` row, not a dead run);
retry+backoff + RPM limiter; result cache/checkpoint → crash-safe, idempotent resume
with no double spend; strict-validation gate option.

## Generalization (no hardcoding)
No test labels, claim IDs, or expected outputs anywhere. Few-shot uses *illustrative*
examples, not copied test answers. Evidence thresholds come from
`evidence_requirements.csv`, not the data.

## Data-scale honesty
Only 20 labeled rows (skew 13/5/2) → **fine-tuning is not viable** (would overfit
`supported`, no signal for `not_enough_information`). We invest in prompt + verification
+ confidence routing, which need no training data. No real train/dev split is claimed.

## Extensibility (intentionally NOT built — would be theater for a CSV deliverable)
- **MCP**: wrap `repository` (history/evidence/images) as MCP tools to make the system
  callable by an interactive adjuster-agent. Clean seam (`repository.py`) already exists.
- **A2A**: the perceiver→verifier→arbiter flow is agent-to-agent *as functions*; a real
  message-bus/transport adds latency and nondeterminism for zero CSV-score gain.
- **A2UI**: `--trace` already emits per-claim `{perception, verifier, engine reasoning,
  row}`; a review UI (image + flags + confidence + override) layers on top read-only.
These are one adapter away each because of the Repository/Provider seams — but out of
scope for a 24h CSV submission.
