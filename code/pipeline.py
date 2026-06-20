"""Per-claim orchestration: load context -> perceive -> decide -> validated row."""

from __future__ import annotations

import engine
import schema
import vision


def _evidence_hint(evidence_rules: list[dict], claim_object: str) -> str:
    """Retrieval: the minimum-image-evidence rules applicable to this object,
    injected into the perception prompt so the model checks the right thing."""
    hints = [r["minimum_image_evidence"] for r in evidence_rules
             if r.get("claim_object") in (claim_object, "all")]
    return " ".join(hints[:4])


def process_claim(provider, claim, history_index, evidence_rules,
                  image_store=None, settings=None) -> dict:
    """Run one claim end-to-end and return a validated 14-column output row.

    perceive -> decide -> validate, via the LLMProvider abstraction. On a
    perception/transport failure, returns a safe not_enough_information row flagged
    for manual review rather than crashing the whole run. Attaches `_usage` and
    `_trace` (stripped before CSV write) for operational analysis and --trace.
    """
    history = history_index.get(claim["user_id"])
    usage = {"input_tokens": 0, "output_tokens": 0}
    trace = {"user_id": claim["user_id"]}
    verifier_on = bool(settings and getattr(settings, "verifier_enabled", False))
    try:
        system = vision.build_system_prompt(settings)
        hint = _evidence_hint(evidence_rules, claim["claim_object"])
        vis = vision.analyze_claim(provider, claim, image_store, system=system, evidence_hint=hint)
        usage = dict(vis["usage"])
        trace["perception"] = vis["analysis"]
        trace["missing_image_ids"] = vis["missing_image_ids"]
        predicted = engine.decide(claim, vis, history, evidence_rules, settings=settings)

        # Adversarial verifier only on the high-stakes 'contradicted' path (cost-bounded).
        if verifier_on and predicted["claim_status"] == "contradicted":
            ci = (vis["analysis"] or {}).get("claim", {})
            verdict, v_usage = vision.verify_claim(
                provider, claim, image_store,
                ci.get("claimed_issue_type", ""), ci.get("claimed_object_part", ""))
            trace["verifier"] = verdict
            usage["input_tokens"] += v_usage.get("input_tokens", 0)
            usage["output_tokens"] += v_usage.get("output_tokens", 0)
            predicted = engine.decide(claim, vis, history, evidence_rules,
                                      settings=settings, verdict=verdict)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, surface in flags
        predicted = _fallback_row(str(exc))
        trace["error"] = str(exc)[:200]

    row = {c: claim[c] for c in schema.PASSTHROUGH_COLUMNS}
    row.update(predicted)
    row = {c: row[c] for c in schema.OUTPUT_COLUMNS}  # enforce column order

    errs = schema.validate_row(row)
    if errs:  # last-resort coercion so output.csv is always contract-valid
        row = _coerce_valid(row)
    row["_usage"] = usage  # stripped before writing; used for operational analysis
    row["_trace"] = trace
    return row


def _fallback_row(reason: str) -> dict:
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": f"Automated review failed: {reason[:120]}",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Could not analyze evidence automatically; flagged for manual review.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


def _coerce_valid(row: dict) -> dict:
    obj = row.get("claim_object", "")
    row["claim_object"] = obj if obj in schema.CLAIM_OBJECTS else "car"
    row["issue_type"] = schema.coerce_enum(row.get("issue_type", ""), schema.ISSUE_TYPES)
    row["object_part"] = schema.normalize_object_part(row.get("object_part", ""), row["claim_object"])
    row["claim_status"] = schema.coerce_enum(row.get("claim_status", ""), schema.CLAIM_STATUSES, "not_enough_information")
    row["severity"] = schema.coerce_enum(row.get("severity", ""), schema.SEVERITIES)
    row["evidence_standard_met"] = schema.bool_str(row.get("evidence_standard_met"))
    row["valid_image"] = schema.bool_str(row.get("valid_image"))
    if row.get("risk_flags") != "none":
        row["risk_flags"] = schema.join_flags(str(row.get("risk_flags", "")).split(";"))
    return row
