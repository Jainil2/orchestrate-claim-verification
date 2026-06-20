"""Offline self-test for the deterministic engine (no API key required).

Feeds synthetic perception into engine.decide and asserts the policy: the three
claim_status paths, "images over history", enum validity, and schema compliance.
Run: python code/test_engine.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine
import schema


def _settings(**kw):
    base = dict(confidence_gating=False, nei_recall_bias=False, strict_issue_match=False,
                verifier_enabled=False)
    base.update(kw)
    return SimpleNamespace(**base)

EVIDENCE_RULES = [
    {"requirement_id": "REQ_GENERAL", "claim_object": "all", "applies_to": "general claim review",
     "minimum_image_evidence": "The claimed object and part should be visible."},
    {"requirement_id": "REQ_CAR_BODY", "claim_object": "car", "applies_to": "dent or scratch",
     "minimum_image_evidence": "The claimed panel should be visible at an angle to assess marks."},
]


def _claim(obj="car", paths="images/test/case_x/img_1.jpg"):
    return {"user_id": "user_x", "claim_object": obj, "image_paths": paths,
            "user_claim": "Customer: there is a dent on the rear bumper."}


def _vis(images, claim_info=None, missing=None):
    return {"analysis": {"claim": claim_info or {"claimed_issue_type": "dent",
            "claimed_object_part": "rear bumper", "claimed_severity": "medium", "ambiguous": False},
            "images": images}, "usage": {}, "loaded_image_ids": ["img_1"],
            "missing_image_ids": missing or []}


def _img(**kw):
    base = {"image_id": "img_1", "object_seen": "car", "matches_claimed_object": True,
            "visible_parts": ["rear_bumper"], "claimed_part_visible": True, "damage": [],
            "image_quality_flags": [], "manipulation_suspected": False,
            "non_original_suspected": False, "text_instruction_present": False, "usable": True}
    base.update(kw)
    return base


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    assert cond, name


def main():
    # 1. SUPPORTED: claimed dent visible on the claimed part
    r = engine.decide(_claim(), _vis([_img(damage=[
        {"issue_type": "dent", "object_part": "rear_bumper", "severity": "medium", "confidence": "high"}])]),
        None, EVIDENCE_RULES)
    check("supported status", r["claim_status"] == "supported")
    check("supported issue_type", r["issue_type"] == "dent")
    check("supported part", r["object_part"] == "rear_bumper")
    check("supported severity", r["severity"] == "medium")
    check("supported cites image", r["supporting_image_ids"] == "img_1")
    check("supported valid row", schema.validate_row({**_claim(), **r}) == [])

    # 2. CONTRADICTED: part clearly visible, no claimed damage present
    r = engine.decide(_claim(), _vis([_img(damage=[])]), None, EVIDENCE_RULES)
    check("contradicted status", r["claim_status"] == "contradicted")
    check("contradicted issue none", r["issue_type"] == "none")
    check("contradicted flags mismatch", "claim_mismatch" in r["risk_flags"])
    check("contradicted valid row", schema.validate_row({**_claim(), **r}) == [])

    # 3. NOT_ENOUGH_INFORMATION: claimed part not visible (wrong angle)
    r = engine.decide(_claim(), _vis([_img(claimed_part_visible=False,
        image_quality_flags=["wrong_angle"], damage=[])]), None, EVIDENCE_RULES)
    check("nei status", r["claim_status"] == "not_enough_information")
    check("nei evidence not met", r["evidence_standard_met"] == "false")
    check("nei support none", r["supporting_image_ids"] == "none")
    check("nei wrong_angle flag", "wrong_angle" in r["risk_flags"])

    # 4. IMAGES OVER HISTORY: high-risk history must NOT flip a clear 'supported'
    hist = {"history_flags": "user_history_risk;manual_review_required"}
    r = engine.decide(_claim(), _vis([_img(damage=[
        {"issue_type": "dent", "object_part": "rear_bumper", "severity": "high", "confidence": "high"}])]),
        hist, EVIDENCE_RULES)
    check("history doesn't flip decision", r["claim_status"] == "supported")
    check("history adds risk flag", "user_history_risk" in r["risk_flags"])
    check("history adds manual review", "manual_review_required" in r["risk_flags"])

    # 5. WRONG OBJECT: image shows a different object
    r = engine.decide(_claim(), _vis([_img(object_seen="laptop", matches_claimed_object=False,
        claimed_part_visible=False, damage=[])]), None, EVIDENCE_RULES)
    check("wrong object -> nei", r["claim_status"] == "not_enough_information")
    check("wrong object flag", "wrong_object" in r["risk_flags"])
    check("wrong object invalid_image", r["valid_image"] == "false")

    # 6. CONFIDENCE GATING: low-confidence support + nei_recall_bias -> NEI + manual review
    low_conf_vis = _vis([_img(damage=[
        {"issue_type": "dent", "object_part": "rear_bumper", "severity": "medium", "confidence": "low"}])])
    r = engine.decide(_claim(), low_conf_vis, None, EVIDENCE_RULES,
                      settings=_settings(confidence_gating=True, nei_recall_bias=True))
    check("low-confidence -> NEI", r["claim_status"] == "not_enough_information")
    check("low-confidence -> manual review", "manual_review_required" in r["risk_flags"])
    # same input, gating OFF -> supported
    r = engine.decide(_claim(), low_conf_vis, None, EVIDENCE_RULES, settings=_settings())
    check("gating off -> supported", r["claim_status"] == "supported")

    # 7. STRICT MATCH: claimed scratch, visible dent (same 'surface' family)
    scratch_claim = _vis([_img(damage=[
        {"issue_type": "dent", "object_part": "rear_bumper", "severity": "low", "confidence": "high"}])],
        claim_info={"claimed_issue_type": "scratch", "claimed_object_part": "rear bumper",
                    "claimed_severity": "low", "ambiguous": False})
    r = engine.decide(_claim(), scratch_claim, None, EVIDENCE_RULES, settings=_settings(strict_issue_match=False))
    check("lenient family match -> supported", r["claim_status"] == "supported")
    r = engine.decide(_claim(), scratch_claim, None, EVIDENCE_RULES, settings=_settings(strict_issue_match=True))
    check("strict match -> not supported", r["claim_status"] != "supported")

    # 8. ARBITER: primary contradicted, verifier says damage IS present -> NEI, not contradicted
    contradicted_vis = _vis([_img(damage=[])])
    r = engine.decide(_claim(), contradicted_vis, None, EVIDENCE_RULES,
                      settings=_settings(verifier_enabled=True),
                      verdict={"damage_present": True, "confidence": "high"})
    check("arbiter overturns contradicted -> NEI", r["claim_status"] == "not_enough_information")
    check("arbiter -> manual review", "manual_review_required" in r["risk_flags"])
    check("arbiter clears claim_mismatch", "claim_mismatch" not in r["risk_flags"])
    check("arbiter row valid", schema.validate_row({**_claim(), **r}) == [])

    # 9. GUARDED consistency override (option 2): exaggerated flag flips supported->
    #    contradicted ONLY with corroboration (history risk OR severity gap).
    def _exag(claimed_sev, observed_sev):
        return _vis([_img(damage=[{"issue_type": "dent", "object_part": "rear_bumper",
                                   "severity": observed_sev, "confidence": "high"}])],
                    claim_info={"claimed_issue_type": "dent", "claimed_object_part": "rear bumper",
                                "claimed_severity": claimed_sev, "ambiguous": False,
                                "evidence_vs_claim": "exaggerated_or_mismatched"})
    hist = {"history_flags": "user_history_risk"}
    # exaggerated + history risk -> contradicted
    r = engine.decide(_claim(), _exag("medium", "medium"), hist, EVIDENCE_RULES, settings=_settings())
    check("exaggerated + history risk -> contradicted", r["claim_status"] == "contradicted")
    # exaggerated + severity gap (claimed high, observed low) -> contradicted (no history)
    r = engine.decide(_claim(), _exag("high", "low"), None, EVIDENCE_RULES, settings=_settings())
    check("exaggerated + severity gap -> contradicted", r["claim_status"] == "contradicted")
    # exaggerated but NO corroboration -> stays supported (guard holds)
    r = engine.decide(_claim(), _exag("medium", "medium"), None, EVIDENCE_RULES, settings=_settings())
    check("exaggerated w/o corroboration -> stays supported", r["claim_status"] == "supported")

    print("\nAll engine self-tests passed.")


if __name__ == "__main__":
    main()
