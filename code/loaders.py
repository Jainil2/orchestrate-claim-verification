"""Dataset loaders: claims, user history, evidence requirements, images.

All paths resolve relative to the repo's dataset/ dir so the system runs from a
clean checkout. Images are read as raw bytes + MIME type (provider-neutral); the
vision layer wraps them in Gemini image parts.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_claims(name: str) -> list[dict]:
    """Load claims.csv or sample_claims.csv (by filename) from dataset/."""
    return read_csv(DATASET_DIR / name)


def load_user_history() -> dict[str, dict]:
    """Index user_history.csv by user_id."""
    rows = read_csv(DATASET_DIR / "user_history.csv")
    return {r["user_id"]: r for r in rows}


def load_evidence_requirements() -> list[dict]:
    return read_csv(DATASET_DIR / "evidence_requirements.csv")


def image_id(path: str) -> str:
    """Image ID = filename without extension (e.g. images/.../img_1.jpg -> img_1)."""
    return Path(path).stem


def parse_image_paths(image_paths: str) -> list[str]:
    return [p.strip() for p in image_paths.split(";") if p.strip()]


def read_image(rel_path: str) -> tuple[bytes, str] | None:
    """Return (raw_bytes, mime_type) for a dataset-relative image path.

    Returns None (caller treats as a missing/unreadable image) rather than raising,
    so one bad path doesn't sink a whole claim.
    """
    full = DATASET_DIR / rel_path
    if not full.exists():
        return None
    media_type = _MEDIA_TYPES.get(full.suffix.lower())
    if media_type is None:
        return None
    return full.read_bytes(), media_type
