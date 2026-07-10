"""ANN index and bundle lifecycle utilities."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learning_to_rank_distillation.models.student import TwoTowerStudent
from learning_to_rank_distillation.models.teacher import content_hash
from learning_to_rank_distillation.production.bundle import (
    load_student_bundle,
    save_student_bundle,
)
from learning_to_rank_distillation.schema import RankingExample


@dataclass(frozen=True, slots=True)
class BundleBuildResult:
    version_dir: Path
    item_count: int
    data_hash: str


def build_student_bundle_version(
    student: TwoTowerStudent,
    examples: list[RankingExample],
    registry_dir: Path,
    *,
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
) -> BundleBuildResult:
    """Build, validate, and stage a versioned student bundle."""

    registry_dir.mkdir(parents=True, exist_ok=True)
    data_hash = content_hash(examples)
    version_dir = registry_dir / f"student-{data_hash[:12]}"
    if version_dir.exists():
        validate_student_bundle(version_dir)
        bundle = load_student_bundle(version_dir)
        return BundleBuildResult(
            version_dir=version_dir, item_count=len(bundle.item_ids), data_hash=data_hash
        )

    staging_dir = registry_dir / f".staging-{data_hash[:12]}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    save_student_bundle(
        student,
        examples,
        staging_dir,
        metrics=metrics,
        training_config=training_config,
    )
    validate_student_bundle(staging_dir)
    staging_dir.rename(version_dir)
    bundle = load_student_bundle(version_dir)
    return BundleBuildResult(
        version_dir=version_dir, item_count=len(bundle.item_ids), data_hash=data_hash
    )


def validate_student_bundle(bundle_dir: Path) -> None:
    """Validate that a bundle can be loaded and searched."""

    bundle = load_student_bundle(bundle_dir)
    if bundle.item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must be a 2D array")
    if len(bundle.item_ids) != bundle.item_embeddings.shape[0]:
        raise ValueError("item ids and embedding rows must align")
    if bundle.item_embeddings.shape[0] == 0:
        raise ValueError("bundle must contain at least one item")
    index = bundle.item_index()
    first_query = bundle.item_embeddings[:1]
    if not index.search(first_query, k=1):
        raise ValueError("bundle index search returned no results")


def publish_bundle_version(registry_dir: Path, version_dir: Path) -> Path:
    """Atomically publish a bundle version by updating the CURRENT pointer."""

    registry_dir.mkdir(parents=True, exist_ok=True)
    if not version_dir.exists():
        raise FileNotFoundError(f"bundle version does not exist: {version_dir}")
    current_path = registry_dir / "CURRENT"
    temporary_path = registry_dir / "CURRENT.tmp"
    temporary_path.write_text(str(version_dir.resolve()), encoding="utf-8")
    os.replace(temporary_path, current_path)
    return current_path


def resolve_published_bundle(registry_dir: Path) -> Path:
    """Resolve the currently published bundle path."""

    current_path = registry_dir / "CURRENT"
    if not current_path.exists():
        raise FileNotFoundError(f"no published bundle pointer at {current_path}")
    return Path(current_path.read_text(encoding="utf-8").strip())
