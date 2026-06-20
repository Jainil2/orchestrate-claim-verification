"""Deterministic decision engine.

Turns the VLM's per-image perception (+ user history + evidence rules) into the
10 predicted output fields. ALL policy lives here, in plain code — no model calls.
This is what makes the system reproducible and auditable, and it enforces the core
rule: images are the source of truth; user history adds risk context but NEVER
flips a clear visual decision.
"""

from __future__ import annotations

import schema

_SEV_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 0}

# Issue families used to pick the relevant evidence_requirements rule.
_FAMILY = {
    "dent": "surface", "scratch": "surface", "stain": "surface", "water_damage": "surface",
    "crack": "structural", "glass_shatter": "structural", "broken_part": "structural",
    "missing_part": "structural", "torn_packaging": "packaging",
    "crushed_packaging": "packaging",
}


_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def _matches(a: str, b: str, strict: bool = False) -> bool:
    """Issue-type match: exact always; same family (dent~scratch, crack~broken)
    only when not strict. `strict` (Settings.strict_issue_match) requires exact."""
    if a == b:
        return True
    if strict:
        return False
    fa, fb = _FAMILY.get(a), _FAMILY.get(b)
    return fa is not None and fa == fb


def _select_evidence_reason(claim_object, issue_type, evidence_rules) -> str:
    """Pick the most specific minimum_image_evidence text for this claim."""
    best_generic = ""
    for r in evidence_rules:
        ro = r["claim_object"]
        if ro == claim_object and issue_type and issue_type in r["applies_to"].replace(",", " "):
            return r["minimum_image_evidence"]
        if ro == claim_object and not best_generic:
            best_generic = r["minimum_image_evidence"]
        if ro == "all" and not best_generic:
            best_generic = r["minimum_image_evidence"]
    return best_generic or "The claimed object and relevant part should be clearly visible."


def decide(claim: dict, vis: dict, history: dict | None, evidence_rules: list[dict],
           settings=None, verdict: dict | None = None) -> dict:
    """Produce the 10 predicted fields. `vis` is vision.analyze_claim()'s result.

    `settings` (optional) toggles confidence-gating / strict matching / NEI bias;
    `verdict` (optional) is an adversarial verifier result for arbiter logic.
    Both default to None so existing callers and tests are unaffected.
    """
    claim_object = claim["claim_object"]
    analysis = vis.get("analysis") or {}
    claim_info = analysis.get("claim") or {}
    images = analysis.get("images") or []

    claimed_issue = schema.coerce_enum(claim_info.get("claimed_issue_type", ""), schema.ISSUE_TYPES)
    claimed_part = schema.normalize_object_part(claim_info.get("claimed_object_part", ""), claim_object)

    usable_imgs = [im for im in images if im.get("usable")]
    obj_match_imgs = [im for im in usable_imgs if im.get("matches_claimed_object")]
    part_visible_imgs = [im for im in obj_match_imgs if im.get("claimed_part_visible")]

    risk = set()

    # --- valid_image: at least one usable image showing the claimed object ---
    valid_image = bool(obj_match_imgs)
    if usable_imgs and not obj_match_imgs:
        risk.add("wrong_object")
    if not usable_imgs:
        valid_image = False

    # --- evidence_standard_met: claimed part inspectable on a valid image ---
    evidence_met = bool(part_visible_imgs)
    evidence_reason = _select_evidence_reason(claim_object, claimed_issue, evidence_rules)
    if not evidence_met:
        if obj_match_imgs and not part_visible_imgs:
            risk.add("wrong_object_part")
        evidence_reason = f"Not met: {evidence_reason}"
    else:
        evidence_reason = f"Met: {evidence_reason}"

    # --- gather matching damage on part-visible images (images are source of truth) ---
    matching_damage = []  # (image_id, damage_dict)
    strict = bool(getattr(settings, "strict_issue_match", False))
    any_damage = []
    for im in obj_match_imgs:
        for d in im.get("damage", []):
            d_issue = schema.coerce_enum(d.get("issue_type", ""), schema.ISSUE_TYPES, "unknown")
            any_damage.append((im.get("image_id"), d_issue, d))
            if _matches(d_issue, claimed_issue, strict) and im.get("claimed_part_visible"):
                matching_damage.append((im.get("image_id"), d))

    # --- claim_status decision (deterministic) ---
    if not valid_image or not evidence_met:
        claim_status = "not_enough_information"
        if obj_match_imgs and not matching_damage:
            risk.add("damage_not_visible")
    elif matching_damage:
        claim_status = "supported"
    else:
        # part is visible and inspectable but the claimed damage is not present ->
        # the visual evidence conflicts with the claim.
        claim_status = "contradicted"
        risk.add("claim_mismatch")
        risk.add("damage_not_visible")

    # --- guarded claim-consistency override ---
    # The model's holistic "exaggerated/mismatched" read alone is too trigger-happy
    # (it broke genuine supported cases). Only flip supported -> contradicted when it
    # is CORROBORATED by a structured signal: the user has a history-risk flag, OR the
    # claimed severity far exceeds what's actually visible (a tell-tale of exaggeration).
    if claim_status == "supported" and valid_image and evidence_met:
        assessment = str(claim_info.get("evidence_vs_claim", "")).strip().lower()
        if assessment == "exaggerated_or_mismatched":
            hist_risk = "user_history_risk" in ((history or {}).get("history_flags") or "")
            claimed_sev = schema.coerce_enum(claim_info.get("claimed_severity", ""), schema.SEVERITIES)
            observed_sev = max((schema.coerce_enum(d.get("severity", ""), schema.SEVERITIES, "none")
                                for _, d in matching_damage), key=lambda s: _SEV_RANK[s], default="none")
            severity_gap = _SEV_RANK[claimed_sev] - _SEV_RANK[observed_sev] >= 2
            if hist_risk or severity_gap:
                claim_status = "contradicted"
                risk.add("claim_mismatch")

    # --- issue_type / object_part (final, image-grounded) ---
    if claim_status == "supported":
        d = matching_damage[0][1]
        issue_type = schema.coerce_enum(d.get("issue_type", ""), schema.ISSUE_TYPES, claimed_issue or "unknown")
        object_part = schema.normalize_object_part(d.get("object_part", "") or claimed_part, claim_object)
    elif claim_status == "contradicted":
        # part visible, no claimed damage -> nothing wrong with that part
        issue_type = "none"
        object_part = claimed_part
    else:  # not_enough_information
        issue_type = "unknown"
        object_part = claimed_part if claimed_part != "unknown" and part_visible_imgs else "unknown"

    # --- severity ---
    if claim_status == "supported":
        sev = max(
            (schema.coerce_enum(d.get("severity", ""), schema.SEVERITIES, "unknown") for _, d in matching_damage),
            key=lambda s: _SEV_RANK[s],
        )
        severity = sev if sev != "unknown" else "low"
    elif claim_status == "contradicted":
        severity = "none"
    else:
        severity = "unknown"

    # --- supporting_image_ids ---
    if claim_status == "supported":
        support = [iid for iid, _ in matching_damage]
    elif claim_status == "contradicted":
        support = [im.get("image_id") for im in part_visible_imgs]
    else:
        support = []
    # dedup, keep order
    seen = set()
    supporting = [s for s in support if s and not (s in seen or seen.add(s))]

    # --- risk flags: image quality + integrity (from perception) ---
    for im in images:
        for q in im.get("image_quality_flags", []):
            if q in schema.RISK_FLAGS:
                risk.add(q)
        if im.get("manipulation_suspected"):
            risk.add("possible_manipulation")
        if im.get("non_original_suspected"):
            risk.add("non_original_image")
        if im.get("text_instruction_present"):
            risk.add("text_instruction_present")
    if claim_info.get("text_instruction_present"):
        risk.add("text_instruction_present")
    if vis.get("missing_image_ids"):
        risk.add("manual_review_required")

    # --- user history: adds risk context ONLY, never flips the visual decision ---
    if history:
        hist_flags = (history.get("history_flags") or "").strip()
        if hist_flags and hist_flags != "none":
            for f in hist_flags.split(";"):
                f = f.strip()
                if f in schema.RISK_FLAGS:
                    risk.add(f)

    # --- confidence gating (uses the per-damage confidence the model emits) ---
    confidence_gating = bool(getattr(settings, "confidence_gating", False))
    nei_recall_bias = bool(getattr(settings, "nei_recall_bias", False))
    if confidence_gating and claim_status == "supported" and matching_damage:
        best_conf = max(_CONF_RANK.get(str(d.get("confidence", "")).lower(), 0)
                        for _, d in matching_damage)
        if best_conf == 0:  # only low-confidence support for asserting the damage
            risk.add("manual_review_required")
            if nei_recall_bias:  # don't assert a low-confidence positive; route to review
                claim_status = "not_enough_information"
                issue_type, severity = "unknown", "unknown"
                supporting = []

    # --- arbiter: reconcile an adversarial verifier verdict (if provided) ---
    # verdict = {"damage_present": bool, "confidence": "low|medium|high"} from a
    # second independent look. Disagreement never silently asserts; it routes to
    # manual review and refuses to call the user wrong on contested evidence.
    if verdict:
        v_present = bool(verdict.get("damage_present"))
        if claim_status == "supported" and not v_present:
            risk.add("manual_review_required")  # verifier doubts the damage
        elif claim_status == "contradicted" and v_present:
            # verifier sees the claimed damage the primary missed -> don't contradict
            claim_status = "not_enough_information"
            issue_type, severity = "unknown", "unknown"
            supporting = []
            risk.add("manual_review_required")
            risk.discard("claim_mismatch")

    # --- manual_review_required: contradiction / integrity / ambiguity ---
    if (claim_status == "contradicted" or "possible_manipulation" in risk
            or "non_original_image" in risk or "user_history_risk" in risk
            or claim_info.get("ambiguous")):
        risk.add("manual_review_required")

    # --- justification ---
    justification = _justify(claim_status, issue_type, object_part, supporting, matching_damage, obj_match_imgs)

    return {
        "evidence_standard_met": schema.bool_str(evidence_met),
        "evidence_standard_met_reason": evidence_reason[:300],
        "risk_flags": schema.join_flags(risk),
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": justification[:400],
        "supporting_image_ids": ";".join(supporting) if supporting else "none",
        "valid_image": schema.bool_str(valid_image),
        "severity": severity,
    }


def _justify(status, issue_type, object_part, supporting, matching_damage, obj_match_imgs) -> str:
    imgs = ", ".join(supporting) if supporting else "the submitted images"
    if status == "supported":
        return (f"Image evidence ({imgs}) shows {issue_type} on the {object_part}, "
                f"matching the claim.")
    if status == "contradicted":
        return (f"The {object_part} is clearly visible in {imgs} but shows no sign of the "
                f"claimed damage; visual evidence conflicts with the claim.")
    if obj_match_imgs:
        return ("The claimed object is shown but the relevant part/damage is not "
                "inspectable from the submitted images; evidence is insufficient.")
    return "The submitted images do not provide usable evidence of the claimed object and damage."
