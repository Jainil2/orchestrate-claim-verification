"""Repository seam over the dataset (Repository pattern).

Decouples the pipeline from the concrete CSV/local-image layout so the source
could later be a DB / object store without touching business logic. Thin wrappers
that reuse the existing parsing in loaders.py — no logic duplicated.
"""

from __future__ import annotations

from pathlib import Path

import loaders

REPO_ROOT = Path(__file__).resolve().parent.parent


class ClaimsRepository:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    def load(self, name: str) -> list[dict]:
        return loaders.read_csv(self.dataset_dir / name)


class HistoryRepository:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    def index(self) -> dict[str, dict]:
        rows = loaders.read_csv(self.dataset_dir / "user_history.csv")
        return {r["user_id"]: r for r in rows}


class EvidenceRepository:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    def rules(self) -> list[dict]:
        return loaders.read_csv(self.dataset_dir / "evidence_requirements.csv")


class ImageStore:
    """Resolves dataset-relative image paths to (bytes, mime). Honors dataset_dir."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    def read(self, rel_path: str) -> tuple[bytes, str] | None:
        full = self.dataset_dir / rel_path
        if not full.exists():
            return None
        mime = loaders._MEDIA_TYPES.get(full.suffix.lower())
        if mime is None:
            return None
        return full.read_bytes(), mime

    @staticmethod
    def ids(image_paths: str) -> list[str]:
        return [loaders.image_id(p) for p in loaders.parse_image_paths(image_paths)]


class Dataset:
    """Bundle of repositories built from one dataset directory."""

    def __init__(self, dataset_dir: Path):
        self.claims = ClaimsRepository(dataset_dir)
        self.history = HistoryRepository(dataset_dir)
        self.evidence = EvidenceRepository(dataset_dir)
        self.images = ImageStore(dataset_dir)
