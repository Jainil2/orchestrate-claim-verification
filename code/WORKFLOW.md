# Workflow — test → improve → finalize → submit

How to operate this system, framed like an ML train/test cycle.

## Mental model (read first)

| ML concept | Here |
|---|---|
| Training set | **None.** 20 labeled rows is far too few to fine-tune (and skewed 13/5/2). There is no training step. |
| Validation / dev set | **`dataset/sample_claims.csv`** — 20 rows that include the gold labels. You tune against this. |
| Test set | **`dataset/claims.csv`** — 44 rows, **no labels** → your `output.csv`. Predict once; **never tune on it**. |
| "Hyperparameters" | Prompt text (`code/vision.py`: `SYSTEM_PROMPT`, `FEW_SHOT`), `engine.py` rules, and `.env` flags: `VERIFIER_ENABLED`, `FEW_SHOT_ENABLED`, `CONFIDENCE_GATING`, `STRICT_ISSUE_MATCH`, `NEI_RECALL_BIAS`. |
| Training loop | **Eval-driven tuning** on the validation set — measure, change one thing, re-measure. No gradient training. |

> **Two cautions.** (1) Only 20 validation rows — improve **macro-F1** *and* keep the reasoning sound; don't overfit row-by-row. (2) `claims.csv` has no labels by design — tuning "to it" is impossible and would just be guessing.

All commands have a `make` shortcut (see `make help`). Use your venv: `make <target> PY=/path/to/venv/bin/python`.

---

## Phase 0 — setup (once)
```bash
make install                      # pip install -r code/requirements.txt
echo "GEMINI_API_KEY=..." > .env  # or your provider's key (see code/README.md)
make test                         # offline engine self-test (no key) — sanity gate
```

## Phase 1 — baseline measure (validation set)
```bash
make eval                         # code/evaluation/main.py --ablations  -> evaluation_report.md
```
Open `code/evaluation/evaluation_report.md` and read, in order:
- **claim_status accuracy + macro-F1** — the primary metric (classes are imbalanced, so watch macro-F1, not just accuracy).
- **confusion matrix** — where supported/contradicted/not_enough_information get confused.
- **per-field error analysis** — which exact rows missed on which field.
- **A vs B** — monolithic baseline vs the hybrid (B should win on reproducibility/clarity).
- **ablation table** — ±verifier / ±few-shot / ±confidence-gating: which levers actually help.

## Phase 2 — improve (the feedback loop, repeat)
1. **One hypothesis** from the error analysis. e.g. *"contradicted false-positives → keep the verifier"*, *"issue_type errors on packages → add a package example to FEW_SHOT"*, *"too many manual_review → loosen confidence gating"*.
2. **Change ONE thing**: a flag in `.env`, prompt text in `code/vision.py`, or a rule in `code/engine.py`.
3. **Re-measure**: `make eval-quick` (5 rows, cheap) → then `make eval` (full 20).
   - The **eval harness does not use the result cache**, so prompt/flag edits are reflected immediately — no version bump needed during tuning.
4. **Keep or revert**: keep only if macro-F1 (and the field you targeted) improved *without* regressing others. Otherwise revert.
5. Repeat until metrics plateau or you run out of time. **Lock the winning `.env` flag values.**

After any change to `engine.py`, re-run `make test` (the deterministic logic must stay green).

## Phase 3 — smoke-test on the real test set
```bash
make smoke                        # code/main.py --limit 3  (3 test claims; confirms it runs + shows cost)
```

## Phase 4 — finalize (full test set → submission output)
```bash
# IMPORTANT cache note: main.py caches results by config + PROMPT_VERSION (eval does NOT cache).
# If you edited PROMPT TEXT since your last main.py run, bump the version so the cache is not stale:
echo "PROMPT_VERSION=v2" >> .env   # change the value whenever prompt text changed
make run                           # code/main.py -> output.csv (44 rows)
#   or force a clean recompute:    make run-fresh   (python code/main.py --no-cache)
```
`make run` must print **"All 44 rows valid against the output contract."** and a cost line.
Optional spot-check: `make trace` then inspect a few files in `code/logs/trace/`
(`{perception, verifier, engine reasoning, final row}` per claim).

## Phase 5 — submit
```bash
make package                       # -> code.zip (excludes .cache/, logs/, __pycache__, .env)
```
Submit three things (AGENTS.md §6 / problem_statement.md):
1. **`output.csv`** — repo root, 44 rows.
2. **`code.zip`** — the runnable solution.
3. **`chat_transcript`** — `$HOME/hackerrank_orchestrate/log.txt`.

---

## Quick reference
| Goal | Command |
|---|---|
| Sanity (no key) | `make test` |
| Measure on validation | `make eval` |
| Fast iterate | `make eval-quick` |
| Try the real test set | `make smoke` |
| Produce submission output | `make run` (or `make run-fresh` after prompt edits) |
| Inspect reasoning | `make trace` |
| Build submission zip | `make package` |
| Reset artifacts | `make clean` |
