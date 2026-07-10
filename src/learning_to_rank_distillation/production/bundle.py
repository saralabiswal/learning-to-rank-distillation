"""Student model artifact bundle format."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learning_to_rank_distillation.models.ann import FaissItemIndex
from learning_to_rank_distillation.models.student import (
    StudentConfig,
    TwoTowerRanker,
    TwoTowerStudent,
)
from learning_to_rank_distillation.models.teacher import content_hash
from learning_to_rank_distillation.schema import RankingExample


@dataclass(frozen=True, slots=True)
class StudentBundle:
    """Loaded student bundle with retrieval-ready item embeddings."""

    path: Path
    student: TwoTowerStudent
    item_ids: list[str]
    item_embeddings: np.ndarray
    item_metadata: list[dict[str, Any]]
    metadata: dict[str, Any]

    def item_index(self) -> FaissItemIndex:
        index = None
        embeddings = np.ascontiguousarray(self.item_embeddings, dtype=np.float32)
        try:
            import faiss

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
        except Exception:
            index = None
        return FaissItemIndex(
            item_ids=self.item_ids,
            index=index,
            embeddings=embeddings,
        )


def save_student_bundle(
    student: TwoTowerStudent,
    item_examples: list[RankingExample],
    output_dir: Path,
    *,
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
) -> Path:
    """Save a coherent student serving bundle."""

    if student.model is None:
        raise RuntimeError("student must be trained or initialized before saving a bundle")

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_items = _unique_items(item_examples)
    item_embeddings = student.item_embeddings(unique_items)
    metadata = {
        "bundle_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_type": "TwoTowerStudent",
        "student_config": asdict(student.config),
        "data_hash": content_hash(item_examples),
        "item_count": len(unique_items),
        "query_feature_names": student.vectorizer.query_feature_names,
        "item_feature_names": student.vectorizer.item_feature_names,
        "query_dim": student.vectorizer.query_dim(),
        "item_dim": student.vectorizer.item_dim(),
        "embedding_dim": student.config.embedding_dim,
        "metrics": metrics or {},
        "training_config": training_config or {},
    }

    torch.save(student.model.state_dict(), output_dir / "student.pt")
    (output_dir / "vectorizer.pkl").write_bytes(pickle.dumps(student.vectorizer))
    np.save(output_dir / "item_embeddings.npy", item_embeddings.astype(np.float32))
    (output_dir / "items.json").write_text(
        json.dumps(
            [
                {
                    "item_id": example.item_id,
                    "group_id": example.group_id,
                    "features": example.features,
                }
                for example in unique_items
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def load_student_bundle(path: Path) -> StudentBundle:
    """Load a student serving bundle from disk."""

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    vectorizer = pickle.loads((path / "vectorizer.pkl").read_bytes())
    config = StudentConfig(**metadata["student_config"])
    student = TwoTowerStudent(config=config, vectorizer=vectorizer)
    student.model = TwoTowerRanker(
        query_dim=vectorizer.query_dim(),
        item_dim=vectorizer.item_dim(),
        config=config,
    )
    student.model.load_state_dict(torch.load(path / "student.pt", map_location="cpu"))
    student.model.eval()
    item_embeddings = np.load(path / "item_embeddings.npy").astype(np.float32)
    item_metadata = json.loads((path / "items.json").read_text(encoding="utf-8"))
    return StudentBundle(
        path=path,
        student=student,
        item_ids=[item["item_id"] for item in item_metadata],
        item_embeddings=item_embeddings,
        item_metadata=item_metadata,
        metadata=metadata,
    )


def _unique_items(examples: list[RankingExample]) -> list[RankingExample]:
    seen: set[str] = set()
    unique: list[RankingExample] = []
    for example in examples:
        if example.item_id not in seen:
            seen.add(example.item_id)
            unique.append(example)
    return unique
