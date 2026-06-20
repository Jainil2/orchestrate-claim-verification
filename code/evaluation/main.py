"""Evaluation harness: score the system on labeled dataset/sample_claims.csv.

Compares strategies (problem_statement.md requires >=2) and runs ablations:
  A (baseline) : one monolithic provider call returns all 10 predicted fields.
  B (hybrid)   : provider perception + deterministic engine (the shipped system).
  Ablations on B: ±verifier, ±few-shot, ±confidence-gating.

All model access goes through the LLMProvider abstraction (no vendor SDK here;
no duplicated call path). Pricing comes from Settings. Writes
code/evaluation/evaluation_report.md.

Usage:
    python code/evaluation/main.py             # A + B + ablations on all 20 sample rows
    python code/evaluation/main.py --limit 5   # quick check
    python code/evaluation/main.py --strategy B
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import schema  # noqa: E402
import vision  # noqa: E402
from config import load_settings  # noqa: E402
from providers import get_provider  # noqa: E402
from providers.base import text_part  # noqa: E402
from repository import Dataset  # noqa: E402
from resilience import ResilientProvider  # noqa: E402
from pipeline import process_claim  # noqa: E402
import scorer  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "evaluation_report.md"

# ---- Baseline A: monolithic single-call strategy (model owns the policy) ----
BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_standard_met": {"type": "boolean"},
        "evidence_standard_met_reason": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "issue_type": {"type": "string"},
        "object_part": {"type": "string"},
        "claim_status": {"type": "string", "description": "supported, contradicted, or not_enough_information"},
        "claim_status_justification": {"type": "string"},
        "supporting_image_ids": {"type": "array", "items": {"type": "string"}},
        "valid_image": {"type": "boolean"},
        "severity": {"type": "string"},
    },
    "required": ["evidence_standard_met", "risk_flags", "issue_type", "object_part",
                 "claim_status", "supporting_image_ids", "valid_image", "severity"],
}

BASELINE_SYSTEM = """You review damage claims (car, laptop, package). Images are the source of truth; the conversation defines the claim; user history adds risk context only and must not override clear visual evidence. Decide supported / contradicted / not_enough_information and return one JSON object matching the schema.
Allowed issue_type: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown.
Allowed claim_status: supported, contradicted, not_enough_information. Allowed severity: none, low, medium, high, unknown."""


def baseline_predict(provider, claim, history, image_store):
    parts, loaded, _ = vision._build_parts(claim, image_store)
    parts.append(text_part(f"user history flags: {(history or {}).get('history_flags', 'none')}"))
    try:
        res = provider.generate_structured(BASELINE_SYSTEM, BASELINE_SCHEMA, parts)
        data, usage = res.analysis, res.usage
    except Exception:  # noqa: BLE001
        return _baseline_fallback(claim), {"input_tokens": 0, "output_tokens": 0}

    obj = claim["claim_object"]
    row = {c: claim[c] for c in schema.PASSTHROUGH_COLUMNS}
    row.update({
        "evidence_standard_met": schema.bool_str(data.get("evidence_standard_met")),
        "evidence_standard_met_reason": str(data.get("evidence_standard_met_reason", ""))[:300],
        "risk_flags": schema.join_flags(data.get("risk_flags", []) or []),
        "issue_type": schema.coerce_enum(data.get("issue_type", ""), schema.ISSUE_TYPES),
        "object_part": schema.normalize_object_part(data.get("object_part", ""), obj),
        "claim_status": schema.coerce_enum(data.get("claim_status", ""), schema.CLAIM_STATUSES, "not_enough_information"),
        "claim_status_justification": str(data.get("claim_status_justification", ""))[:400],
        "supporting_image_ids": ";".join(data.get("supporting_image_ids", []) or []) or "none",
        "valid_image": schema.bool_str(data.get("valid_image")),
        "severity": schema.coerce_enum(data.get("severity", ""), schema.SEVERITIES),
    })
    return {c: row[c] for c in schema.OUTPUT_COLUMNS}, usage


def _baseline_fallback(claim):
    row = {c: claim[c] for c in schema.PASSTHROUGH_COLUMNS}
    row.update({"evidence_standard_met": "false", "evidence_standard_met_reason": "call failed",
                "risk_flags": "manual_review_required", "issue_type": "unknown",
                "object_part": "unknown", "claim_status": "not_enough_information",
                "claim_status_justification": "call failed", "supporting_image_ids": "none",
                "valid_image": "false", "severity": "unknown"})
    return {c: row[c] for c in schema.OUTPUT_COLUMNS}


def run_strategy(name, claims, golds, ds, provider, settings):
    preds, totals = [], {"in": 0, "out": 0}
    history = ds.history.index()
    rules = ds.evidence.rules()
    t0 = time.monotonic()
    for i, claim in enumerate(claims, 1):
        if name == "A":
            row, usage = baseline_predict(provider, claim, history.get(claim["user_id"]), ds.images)
        else:
            row = process_claim(provider, claim, history, rules, ds.images, settings)
            usage = row.pop("_usage", {})
            row = {c: row[c] for c in schema.OUTPUT_COLUMNS}
        totals["in"] += usage.get("input_tokens", 0)
        totals["out"] += usage.get("output_tokens", 0)
        preds.append(row)
        print(f"  [{name}] {i}/{len(claims)} {claim['user_id']} -> {row['claim_status']}", flush=True)
    return {"preds": preds, "metrics": scorer.score_all(preds, golds),
            "tokens": totals, "elapsed": time.monotonic() - t0}


def error_rows(claims, preds, golds, fields):
    out = []
    for claim, p, g in zip(claims, preds, golds):
        diffs = {f: (g.get(f), p.get(f)) for f in fields if p.get(f) != g.get(f)}
        if diffs:
            out.append({"user_id": claim["user_id"], "object": claim["claim_object"], "diffs": diffs})
    return out


def write_report(results, ablations, claims, golds, settings, n_images):
    din, dout = settings.pricing()
    lines = ["# Evaluation Report — Multi-Modal Claim Verification\n",
             f"Provider `{settings.provider}` · model `{settings.resolved_model()}` · "
             f"sample rows {len(golds)} · images {n_images}\n"]

    lines.append("## 1. Strategy comparison (A monolithic vs B hybrid)\n")
    lines.append("| Field (accuracy) | " + " | ".join(results.keys()) + " |")
    lines.append("|---|" + "|".join("---" for _ in results) + "|")
    for f in scorer.CATEGORICAL:
        lines.append(f"| {f} | " + " | ".join(
            f"{results[s]['metrics']['accuracy'][f]:.0%}" for s in results) + " |")
    lines.append("| claim_status macro-F1 | " + " | ".join(
        f"{results[s]['metrics']['claim_status_macro_f1']:.2f}" for s in results) + " |")
    for f in scorer.SET_FIELDS:
        lines.append(f"| {f} (set-F1) | " + " | ".join(
            f"{results[s]['metrics']['set'][f]['f1']:.2f}" for s in results) + " |")
    lines.append("")

    if ablations:
        lines.append("## 2. Ablations on hybrid B (claim_status acc / macro-F1)\n")
        lines.append("| Variant | claim_status acc | macro-F1 |")
        lines.append("|---|---|---|")
        for label, r in ablations.items():
            m = r["metrics"]
            lines.append(f"| {label} | {m['accuracy']['claim_status']:.0%} | {m['claim_status_macro_f1']:.2f} |")
        lines.append("")

    final = "B" if "B" in results else next(iter(results))
    lines.append(f"## 3. claim_status confusion matrix (strategy {final})\n")
    cm = results[final]["metrics"]["claim_status_confusion"]
    statuses = sorted(schema.CLAIM_STATUSES)
    lines.append("| gold \\ pred | " + " | ".join(statuses) + " |")
    lines.append("|---|" + "|".join("---" for _ in statuses) + "|")
    for g in statuses:
        lines.append(f"| {g} | " + " | ".join(str(cm.get(g, {}).get(p, 0)) for p in statuses) + " |")
    lines.append("")

    lines.append(f"## 4. Error analysis (strategy {final}, per-field mismatches)\n")
    errs = error_rows(claims, results[final]["preds"], golds,
                      ["claim_status", "issue_type", "object_part", "severity"])
    if not errs:
        lines.append("No mismatches on the scored fields.\n")
    for e in errs:
        d = "; ".join(f"{k}: gold=`{v[0]}` pred=`{v[1]}`" for k, v in e["diffs"].items())
        lines.append(f"- `{e['user_id']}` ({e['object']}): {d}")
    lines.append("")

    lines.append("## 5. Operational analysis\n")
    lines.append("| Strategy | claims | input tok | output tok | est. cost | runtime |")
    lines.append("|---|---|---|---|---|---|")
    for s, r in results.items():
        t = r["tokens"]
        cost = t["in"] / 1e6 * din + t["out"] / 1e6 * dout
        lines.append(f"| {s} | {len(golds)} | {t['in']:,} | {t['out']:,} | ${cost:.3f} | {r['elapsed']:.0f}s |")
    rb = results.get(final)
    if rb and golds:
        per_in, per_out = rb["tokens"]["in"] / len(golds), rb["tokens"]["out"] / len(golds)
        test_cost = (per_in * 44) / 1e6 * din + (per_out * 44) / 1e6 * dout
        lines.append(f"\n**Projected full test set (44 claims, {final}):** ~{per_in*44:,.0f} in / "
                     f"{per_out*44:,.0f} out tokens, ~${test_cost:.2f} at {settings.provider} pricing "
                     f"(${din}/{dout} per MTok). Hybrid uses 1 call/claim + a 2nd verifier call only on "
                     f"contested ('contradicted') claims; result cache makes re-runs free.")
    lines.append("\n**Rate/cost controls:** temperature=0; app-level retry+backoff + RPM limiter "
                 "(`resilience.py`); per-claim result cache/checkpoint enables idempotent resume; "
                 "bounded concurrency for throughput.\n")

    lines.append("## 6. Notes / honesty\n")
    lines.append("- Only 20 labeled rows (skew 13/5/2) — too few for a real train/dev split or "
                 "fine-tuning; we report on all 20 and invest in prompt + verification + confidence "
                 "routing instead.\n- Strategy B keeps the model on perception and all policy in "
                 "deterministic, tested code (reproducible + auditable). That's why B is shipped.\n")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


ABLATION_VARIANTS = {
    "B (full)": {},
    "B −verifier": {"verifier_enabled": False},
    "B −few-shot": {"few_shot_enabled": False},
    "B −confidence-gating": {"confidence_gating": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--strategy", choices=["A", "B", "both"], default="both")
    ap.add_argument("--ablations", action="store_true", help="run B ablations")
    args = ap.parse_args()

    settings = load_settings()
    ds = Dataset(settings.dataset_dir)
    claims = ds.claims.load("sample_claims.csv")
    if args.limit:
        claims = claims[:args.limit]
    golds = claims  # sample_claims.csv carries labels in the same rows
    n_images = sum(len(ds.images.ids(c["image_paths"])) for c in claims)
    provider = ResilientProvider(get_provider(settings), rpm=settings.rpm, retries=settings.retries)

    strategies = ["A", "B"] if args.strategy == "both" else [args.strategy]
    results = {}
    for s in strategies:
        print(f"Running strategy {s}...")
        results[s] = run_strategy(s, claims, golds, ds, provider, settings)

    ablations = {}
    if args.ablations and "B" in strategies:
        for label, overrides in ABLATION_VARIANTS.items():
            if not overrides:
                ablations[label] = results["B"]
                continue
            print(f"Ablation {label}...")
            variant = dataclasses.replace(settings, **overrides) if dataclasses.is_dataclass(settings) \
                else settings.model_copy(update=overrides)
            ablations[label] = run_strategy("B", claims, golds, ds, provider, variant)

    write_report(results, ablations, claims, golds, settings, n_images)
    for s, r in results.items():
        print(f"Strategy {s}: claim_status acc = {r['metrics']['accuracy']['claim_status']:.0%}, "
              f"macro-F1 = {r['metrics']['claim_status_macro_f1']:.2f}")


if __name__ == "__main__":
    main()
