"""FAISS-backed item retrieval for student embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning_to_rank_distillation.models.student import TwoTowerStudent
from learning_to_rank_distillation.schema import RankingExample


@dataclass(slots=True)
class FaissItemIndex:
    """CPU FAISS index over precomputed item-side embeddings.

    Implements the ANN retrieval portion of FR-2.1.
    """

    item_ids: list[str]
    index: object
    embeddings: np.ndarray

    @classmethod
    def from_student(
        cls,
        student: TwoTowerStudent,
        examples: list[RankingExample],
    ) -> FaissItemIndex:
        import faiss

        unique_examples = _unique_items(examples)
        embeddings = np.ascontiguousarray(
            student.item_embeddings(unique_examples), dtype=np.float32
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls([example.item_id for example in unique_examples], index, embeddings)

    def search(
        self,
        query_embeddings: np.ndarray,
        *,
        k: int,
    ) -> list[list[tuple[str, float]]]:
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        scores = query_embeddings @ self.embeddings.T
        top_k = min(k, len(self.item_ids))
        indices = np.argsort(-scores, axis=1)[:, :top_k]
        distances = np.take_along_axis(scores, indices, axis=1)
        results: list[list[tuple[str, float]]] = []
        for row_distances, row_indices in zip(distances, indices, strict=True):
            results.append(
                [
                    (self.item_ids[int(index)], float(distance))
                    for distance, index in zip(row_distances, row_indices, strict=True)
                    if index >= 0
                ]
            )
        return results


def _unique_items(examples: list[RankingExample]) -> list[RankingExample]:
    seen: set[str] = set()
    unique: list[RankingExample] = []
    for example in examples:
        if example.item_id not in seen:
            seen.add(example.item_id)
            unique.append(example)
    return unique
