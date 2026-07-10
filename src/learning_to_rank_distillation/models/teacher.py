"""Teacher ranking models."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from learning_to_rank_distillation.benchmark.metrics import ndcg_at_k
from learning_to_rank_distillation.features import FeatureVectorizer
from learning_to_rank_distillation.schema import RankingExample, group_by_query, validate_examples


class Teacher(Protocol):
    """Teacher interface required by FR-1.1."""

    def fit(self, examples: list[RankingExample]) -> Teacher:
        """Fit the teacher on ranking examples."""

    def predict(self, examples: list[RankingExample]) -> np.ndarray:
        """Score examples for ranking."""


@dataclass(slots=True)
class LightGBMLambdaMARTTeacher:
    """LightGBM LambdaMART teacher.

    Implements FR-1.1, FR-1.2, FR-1.3, and FR-1.4.
    """

    random_state: int = 13
    n_estimators: int = 40
    learning_rate: float = 0.05
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    model: object | None = None

    def fit(self, examples: list[RankingExample]) -> LightGBMLambdaMARTTeacher:
        import lightgbm as lgb

        ordered_examples, group_sizes = _ordered_examples_and_group_sizes(examples)
        features = self.vectorizer.fit_transform(ordered_examples)
        labels = np.asarray([example.label for example in ordered_examples], dtype=np.float32)
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            deterministic=True,
            force_col_wise=True,
            min_child_samples=1,
            min_data_in_leaf=1,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(features, labels, group=group_sizes)
        return self

    def predict(self, examples: list[RankingExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("teacher must be fitted before predict")
        features = self.vectorizer.transform(examples)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            predictions = self.model.predict(features)
        return np.asarray(predictions, dtype=np.float32)

    def evaluate(self, examples: list[RankingExample]) -> dict[str, float]:
        scores = self.predict(examples)
        return {
            "ndcg@5": ndcg_at_k(examples, scores, k=5),
            "ndcg@10": ndcg_at_k(examples, scores, k=10),
        }

    def save(self, output_dir: Path, training_examples: list[RankingExample]) -> Path:
        if self.model is None:
            raise RuntimeError("teacher must be fitted before save")
        output_dir.mkdir(parents=True, exist_ok=True)
        data_hash = content_hash(training_examples)
        model_path = output_dir / f"teacher-{data_hash[:12]}.txt"
        metadata_path = output_dir / f"teacher-{data_hash[:12]}.json"
        self.model.booster_.save_model(str(model_path))
        metadata_path.write_text(
            json.dumps(
                {
                    "model_type": "LightGBMLambdaMARTTeacher",
                    "data_hash": data_hash,
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "random_state": self.random_state,
                    "feature_dim": self.vectorizer.output_dim(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return metadata_path


def content_hash(examples: list[RankingExample]) -> str:
    """Create a stable content hash for training examples."""

    validate_examples(examples)
    digest = hashlib.sha256()
    for example in examples:
        digest.update(
            json.dumps(
                {
                    "query_id": example.query_id,
                    "item_id": example.item_id,
                    "group_id": example.group_id,
                    "label": example.label,
                    "is_unbiased": example.is_unbiased,
                    "position": example.position,
                    "features": example.features,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _ordered_examples_and_group_sizes(
    examples: list[RankingExample],
) -> tuple[list[RankingExample], list[int]]:
    grouped = group_by_query(examples)
    ordered_examples: list[RankingExample] = []
    group_sizes: list[int] = []
    for query_examples in grouped.values():
        ordered_examples.extend(query_examples)
        group_sizes.append(len(query_examples))
    return ordered_examples, group_sizes
