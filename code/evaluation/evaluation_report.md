# Evaluation Report — Multi-Modal Claim Verification

Provider `gemini` · model `gemini-2.5-flash` · sample rows 20 · images 29

## 1. Strategy comparison (A monolithic vs B hybrid)

| Field (accuracy) | A | B |
|---|---|---|
| claim_status | 75% | 80% |
| issue_type | 35% | 55% |
| object_part | 60% | 60% |
| severity | 30% | 25% |
| evidence_standard_met | 70% | 85% |
| valid_image | 80% | 85% |
| claim_status macro-F1 | 0.66 | 0.64 |
| risk_flags (set-F1) | 0.42 | 0.60 |
| supporting_image_ids (set-F1) | 0.86 | 0.89 |

## 3. claim_status confusion matrix (strategy B)

| gold \ pred | contradicted | not_enough_information | supported |
|---|---|---|---|
| contradicted | 1 | 2 | 2 |
| not_enough_information | 0 | 2 | 0 |
| supported | 0 | 0 | 13 |

## 4. Error analysis (strategy B, per-field mismatches)

- `user_001` (car): object_part: gold=`rear_bumper` pred=`unknown`; severity: gold=`medium` pred=`high`
- `user_002` (car): issue_type: gold=`scratch` pred=`dent`; severity: gold=`low` pred=`high`
- `user_004` (car): severity: gold=`medium` pred=`high`
- `user_007` (car): issue_type: gold=`broken_part` pred=`glass_shatter`; severity: gold=`medium` pred=`high`
- `user_005` (car): issue_type: gold=`scratch` pred=`none`; severity: gold=`low` pred=`none`
- `user_006` (car): object_part: gold=`headlight` pred=`unknown`
- `user_003` (car): severity: gold=`medium` pred=`high`
- `user_008` (car): claim_status: gold=`contradicted` pred=`not_enough_information`; issue_type: gold=`broken_part` pred=`unknown`; object_part: gold=`front_bumper` pred=`unknown`; severity: gold=`high` pred=`unknown`
- `user_009` (laptop): issue_type: gold=`crack` pred=`glass_shatter`; severity: gold=`medium` pred=`high`
- `user_010` (laptop): severity: gold=`medium` pred=`high`
- `user_011` (laptop): issue_type: gold=`stain` pred=`water_damage`; severity: gold=`medium` pred=`high`
- `user_012` (laptop): severity: gold=`low` pred=`medium`
- `user_018` (laptop): issue_type: gold=`crack` pred=`glass_shatter`; severity: gold=`medium` pred=`high`
- `user_020` (laptop): claim_status: gold=`contradicted` pred=`supported`; issue_type: gold=`none` pred=`scratch`; object_part: gold=`trackpad` pred=`unknown`; severity: gold=`none` pred=`low`
- `user_015` (package): object_part: gold=`package_corner` pred=`unknown`
- `user_030` (package): object_part: gold=`seal` pred=`unknown`
- `user_031` (package): object_part: gold=`package_side` pred=`unknown`
- `user_033` (package): claim_status: gold=`contradicted` pred=`not_enough_information`; severity: gold=`low` pred=`unknown`
- `user_034` (package): claim_status: gold=`contradicted` pred=`supported`; issue_type: gold=`none` pred=`torn_packaging`; object_part: gold=`seal` pred=`unknown`; severity: gold=`none` pred=`high`

## 5. Operational analysis

| Strategy | claims | input tok | output tok | est. cost | runtime |
|---|---|---|---|---|---|
| A | 20 | 12,607 | 3,573 | $0.013 | 135s |
| B | 20 | 24,342 | 9,330 | $0.031 | 159s |

**Projected full test set (44 claims, B):** ~53,552 in / 20,526 out tokens, ~$0.07 at gemini pricing ($0.3/2.5 per MTok). Hybrid uses 1 call/claim + a 2nd verifier call only on contested ('contradicted') claims; result cache makes re-runs free.

**Rate/cost controls:** temperature=0; app-level retry+backoff + RPM limiter (`resilience.py`); per-claim result cache/checkpoint enables idempotent resume; bounded concurrency for throughput.

## 6. Notes / honesty

- Only 20 labeled rows (skew 13/5/2) — too few for a real train/dev split or fine-tuning; we report on all 20 and invest in prompt + verification + confidence routing instead.
- Strategy B keeps the model on perception and all policy in deterministic, tested code (reproducible + auditable). That's why B is shipped.
