"""Output schema, allowed enums, and validation.

Single source of truth for the 14-column output contract from problem_statement.md.
The decision engine and validator both import from here so the contract can't drift.
"""

from __future__ import annotations

# --- Output column order (problem_statement.md §"Required output") ---
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

PASSTHROUGH_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]

CLAIM_OBJECTS = {"car", "laptop", "package"}

ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

# object_part is object-specific; "unknown" is valid for all.
OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
        "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents", "item",
        "unknown",
    },
}

CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}

RISK_FLAGS = {
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
}

SEVERITIES = {"none", "low", "medium", "high", "unknown"}


def normalize_object_part(value: str, claim_object: str) -> str:
    """Map a free-text / loosely-typed part to the allowed enum for this object.

    The VLM returns descriptive part names; this snaps them onto the contract.
    Falls back to 'unknown' rather than emitting an invalid value.
    """
    if not value:
        return "unknown"
    allowed = OBJECT_PARTS.get(claim_object, set())
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    # common synonyms -> canonical
    synonyms = {
        "bumper": "rear_bumper" if "rear" in value.lower() else "front_bumper",
        "windscreen": "windshield",
        "mirror": "side_mirror",
        "boot": "body",
        "trunk": "body",
        "panel": "quarter_panel" if claim_object == "car" else "body",
        "display": "screen",
        "keys": "keyboard",
        "track_pad": "trackpad",
        "touchpad": "trackpad",
        "carton": "box",
        "parcel": "box",
        "flap": "seal",
        "sticker": "label",
        "content": "contents",
    }
    if v in synonyms and synonyms[v] in allowed:
        return synonyms[v]
    # substring match against allowed parts (e.g. "rear bumper area" -> rear_bumper)
    for part in allowed:
        if part != "unknown" and part in v:
            return part
    return "unknown"


def coerce_enum(value: str, allowed: set, default: str = "unknown") -> str:
    if not value:
        return default
    v = value.strip().lower()
    return v if v in allowed else default


def bool_str(value) -> str:
    """Render a Python/JSON bool to the canonical 'true'/'false' string."""
    if isinstance(value, str):
        return "true" if value.strip().lower() == "true" else "false"
    return "true" if value else "false"


def join_flags(flags) -> str:
    """Dedup + sort risk flags into a ';'-joined string, or 'none' if empty."""
    clean = sorted({f for f in flags if f in RISK_FLAGS})
    return ";".join(clean) if clean else "none"


def validate_row(row: dict) -> list[str]:
    """Return a list of contract violations for one output row (empty = valid)."""
    errors = []
    for col in OUTPUT_COLUMNS:
        if col not in row:
            errors.append(f"missing column: {col}")
    if errors:
        return errors

    obj = row["claim_object"]
    if obj not in CLAIM_OBJECTS:
        errors.append(f"claim_object invalid: {obj!r}")
    if row["evidence_standard_met"] not in ("true", "false"):
        errors.append(f"evidence_standard_met not bool: {row['evidence_standard_met']!r}")
    if row["valid_image"] not in ("true", "false"):
        errors.append(f"valid_image not bool: {row['valid_image']!r}")
    if row["issue_type"] not in ISSUE_TYPES:
        errors.append(f"issue_type invalid: {row['issue_type']!r}")
    if row["object_part"] not in OBJECT_PARTS.get(obj, set()):
        errors.append(f"object_part invalid for {obj}: {row['object_part']!r}")
    if row["claim_status"] not in CLAIM_STATUSES:
        errors.append(f"claim_status invalid: {row['claim_status']!r}")
    if row["severity"] not in SEVERITIES:
        errors.append(f"severity invalid: {row['severity']!r}")
    flags = row["risk_flags"]
    if flags != "none":
        for f in flags.split(";"):
            if f not in RISK_FLAGS:
                errors.append(f"risk_flag invalid: {f!r}")
    return errors
