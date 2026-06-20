"""Entry point: read dataset/claims.csv -> write output.csv.

Concurrent, resumable, provider-agnostic. Provider/model/behavior come from
config.Settings (env/.env + CLI overrides).

Usage:
    export GEMINI_API_KEY=...                 # default Gemini provider
    python code/main.py                       # full test set -> output.csv
    python code/main.py --limit 3             # smoke test
    PROVIDER=ollama python code/main.py        # local model
    python code/main.py --trace               # dump per-claim reasoning to code/logs/trace/
    python code/main.py --no-cache            # ignore the result store (force recompute)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# allow `python code/main.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema  # noqa: E402
from config import load_settings  # noqa: E402
from providers import get_provider  # noqa: E402
from repository import Dataset  # noqa: E402
from resilience import ResilientProvider, ResultStore  # noqa: E402
from pipeline import process_claim  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("orchestrate")
VOLATILE = ("_usage", "_trace", "_cached")


def _clean(row: dict) -> dict:
    return {c: row[c] for c in schema.OUTPUT_COLUMNS}


def run(settings, input_name: str, output_path: Path, limit: int | None, trace: bool) -> dict:
    ds = Dataset(settings.dataset_dir)
    claims = ds.claims.load(input_name)
    if limit:
        claims = claims[:limit]
    history = ds.history.index()
    rules = ds.evidence.rules()

    provider = ResilientProvider(get_provider(settings), rpm=settings.rpm, retries=settings.retries)
    store = ResultStore(settings.cache_dir, settings, enabled=settings.cache_enabled)

    def work(idx_claim):
        idx, claim = idx_claim
        cached = store.get(claim)
        if cached is not None:
            cached["_cached"] = True
            log.info("claim done", extra={"i": idx, "user": claim["user_id"],
                                          "status": cached.get("claim_status"), "cached": True})
            return idx, cached
        row = process_claim(provider, claim, history, rules, ds.images, settings)
        store.put(claim, row)
        log.info("claim done", extra={"i": idx, "user": claim["user_id"],
                                      "status": row["claim_status"], "cached": False})
        return idx, row

    t0 = time.monotonic()
    results: list = [None] * len(claims)
    with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as ex:
        for idx, row in ex.map(work, enumerate(claims)):
            results[idx] = row
            tag = " (cached)" if row.get("_cached") else ""
            print(f"[{idx + 1}/{len(claims)}] {claims[idx]['user_id']} -> {row['claim_status']}{tag}",
                  flush=True)
    elapsed = time.monotonic() - t0

    # --- aggregate + write (input order preserved) ---
    totals = {"in": 0, "out": 0, "images": 0, "manual_review": 0, "cached": 0}
    clean_rows = []
    if trace:
        trace_dir = REPO_ROOT / "code" / "logs" / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(results):
        u = row.get("_usage", {})
        totals["in"] += u.get("input_tokens", 0)
        totals["out"] += u.get("output_tokens", 0)
        totals["images"] += len(ds.images.ids(claims[idx]["image_paths"]))
        totals["cached"] += int(bool(row.get("_cached")))
        if "manual_review_required" in row["risk_flags"]:
            totals["manual_review"] += 1
        if trace and row.get("_trace") is not None:
            (trace_dir / f"{idx:03d}_{claims[idx]['user_id']}.json").write_text(
                json.dumps({"row": _clean(row), "trace": row["_trace"]}, indent=2), encoding="utf-8")
        clean_rows.append(_clean(row))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(clean_rows)

    bad = [(i, schema.validate_row(r)) for i, r in enumerate(clean_rows) if schema.validate_row(r)]
    if bad:
        msg = f"{len(bad)} rows failed validation: {bad[:3]}"
        if settings.strict_validation:
            raise SystemExit(f"STRICT VALIDATION FAILED: {msg}")
        print(f"WARNING: {msg}", file=sys.stderr)
    else:
        print(f"All {len(clean_rows)} rows valid against the output contract.")

    din, dout = settings.pricing()
    cost = totals["in"] / 1e6 * din + totals["out"] / 1e6 * dout
    print(f"\nProvider: {settings.provider} | model: {settings.resolved_model()} "
          f"| claims: {len(clean_rows)} | cached: {totals['cached']} | images: {totals['images']}")
    print(f"Tokens: {totals['in']:,} in / {totals['out']:,} out | est. cost ${cost:.3f} "
          f"| manual_review: {totals['manual_review']} | runtime: {elapsed:.1f}s "
          f"(concurrency={settings.concurrency})")
    print(f"Wrote {output_path}")
    return {"rows": len(clean_rows), **totals, "elapsed": elapsed, "cost": cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="claims.csv")
    ap.add_argument("--output", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--provider", default=None, help="gemini|openai|ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--trace", action="store_true", help="dump per-claim reasoning")
    ap.add_argument("--no-cache", action="store_true", help="ignore result store / force recompute")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings(provider=args.provider, model=args.model,
                             concurrency=args.concurrency,
                             cache_enabled=False if args.no_cache else None)
    output_path = Path(args.output) if args.output else settings.output_path
    run(settings, args.input, output_path, args.limit, args.trace)


if __name__ == "__main__":
    main()
