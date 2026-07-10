"""Ranking quality metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from learning_to_rank_distillation.schema import RankingExample, group_by_query


def ndcg_at_k(
    examples: list[RankingExample],
    scores: Sequence[float],
    *,
    k: int,
) -> float:
    """Compute mean NDCG@k over query groups.

    Implements FR-1.3.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if len(examples) != len(scores):
        raise ValueError("examples and scores must have the same length")

    score_by_key = {
        (example.query_id, example.item_id): float(score)
        for example, score in zip(examples, scores, strict=True)
    }
    values: list[float] = []
    for query_examples in group_by_query(examples).values():
        labels_by_score = [
            (example.label, score_by_key[(example.query_id, example.item_id)])
            for example in query_examples
        ]
        ranked_pairs = sorted(labels_by_score, key=lambda pair: pair[1], reverse=True)
        predicted = [label for label, _ in ranked_pairs]
        ideal = sorted((label for label, _ in labels_by_score), reverse=True)
        ideal_dcg = _dcg(ideal[:k])
        values.append(0.0 if ideal_dcg == 0.0 else _dcg(predicted[:k]) / ideal_dcg)
    return float(np.mean(values))


def _dcg(labels: Sequence[float]) -> float:
    return sum((2.0**label - 1.0) / math.log2(index + 2) for index, label in enumerate(labels))
