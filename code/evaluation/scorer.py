"""Metrics for scoring predictions against labeled sample_claims.csv.

No metric is specified by the problem, so we define our own: per-field accuracy
on the categorical fields, macro-F1 + confusion matrix for the primary decision
field (claim_status), and set-based scores for the multi-label / multi-id fields.
"""

from __future__ import annotations

from collections import defaultdict

CATEGORICAL = ["claim_status", "issue_type", "object_part", "severity",
               "evidence_standard_met", "valid_image"]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]


def _as_set(value: str) -> set:
    if not value or value.strip().lower() == "none":
        return set()
    return {x.strip() for x in value.split(";") if x.strip()}


def field_accuracy(preds: list[dict], golds: list[dict], field: str) -> float:
    hits = sum(1 for p, g in zip(preds, golds) if p.get(field) == g.get(field))
    return hits / len(golds) if golds else 0.0


def set_metrics(preds: list[dict], golds: list[dict], field: str) -> dict:
    tp = fp = fn = exact = 0
    jacc_sum = 0.0
    for p, g in zip(preds, golds):
        ps, gs = _as_set(p.get(field, "")), _as_set(g.get(field, ""))
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
        exact += int(ps == gs)
        union = ps | gs
        jacc_sum += 1.0 if not union else len(ps & gs) / len(union)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    n = len(golds) or 1
    return {"precision": prec, "recall": rec, "f1": f1,
            "exact_match": exact / n, "jaccard": jacc_sum / n}


def confusion(preds: list[dict], golds: list[dict], field: str) -> dict:
    cm = defaultdict(lambda: defaultdict(int))
    for p, g in zip(preds, golds):
        cm[g.get(field)][p.get(field)] += 1
    return {k: dict(v) for k, v in cm.items()}


def macro_f1(preds: list[dict], golds: list[dict], field: str) -> float:
    labels = {g.get(field) for g in golds} | {p.get(field) for p in preds}
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p.get(field) == lab and g.get(field) == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p.get(field) == lab and g.get(field) != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p.get(field) != lab and g.get(field) == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def score_all(preds: list[dict], golds: list[dict]) -> dict:
    out = {"n": len(golds), "accuracy": {}, "set": {},
           "claim_status_macro_f1": macro_f1(preds, golds, "claim_status"),
           "claim_status_confusion": confusion(preds, golds, "claim_status")}
    for f in CATEGORICAL:
        out["accuracy"][f] = field_accuracy(preds, golds, f)
    for f in SET_FIELDS:
        out["set"][f] = set_metrics(preds, golds, f)
    return out
