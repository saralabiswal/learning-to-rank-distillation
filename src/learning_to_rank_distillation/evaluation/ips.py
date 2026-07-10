"""Inverse propensity scoring for logged ranking evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from learning_to_rank_distillation.schema import RankingExample, group_by_query, validate_examples


def estimate_position_propensities(
    examples: list[RankingExample],
    *,
    min_propensity: float = 0.05,
) -> dict[int, float]:
    """Estimate clipped position propensities from rows with observed positions."""

    validate_examples(examples)
    if not 0.0 < min_propensity <= 1.0:
        raise ValueError("min_propensity must be in (0, 1]")

    counts: dict[int, int] = {}
    for example in examples:
        if example.position is not None:
            counts[example.position] = counts.get(example.position, 0) + 1
    if not counts:
        raise ValueError("at least one example must have an observed position")

    max_count = max(counts.values())
    return {
        position: max(min_propensity, count / max_count)
        for position, count in sorted(counts.items())
    }


def ips_ndcg_at_k(
    examples: list[RankingExample],
    scores: Sequence[float],
    *,
    k: int,
    propensities: dict[int, float] | None = None,
    min_propensity: float = 0.05,
) -> float:
    """Compute inverse-propensity-weighted NDCG@k for logged ranking rows."""

    validate_examples(examples)
    if len(examples) != len(scores):
        raise ValueError("examples and scores must have the same length")
    if k < 1:
        raise ValueError("k must be positive")

    position_propensities = propensities or estimate_position_propensities(
        examples,
        min_propensity=min_propensity,
    )
    grouped_scores = {
        (example.query_id, example.item_id): float(score)
        for example, score in zip(examples, scores, strict=True)
    }
    values: list[float] = []
    for query_examples in group_by_query(examples).values():
        ranked = sorted(
            query_examples,
            key=lambda example: grouped_scores[(example.query_id, example.item_id)],
            reverse=True,
        )
        dcg = _weighted_dcg(ranked, position_propensities, k=k, min_propensity=min_propensity)
        ideal = _weighted_dcg(
            sorted(
                query_examples,
                key=lambda example: _weighted_gain(
                    example,
                    position_propensities,
                    min_propensity=min_propensity,
                ),
                reverse=True,
            ),
            position_propensities,
            k=k,
            min_propensity=min_propensity,
        )
        if ideal > 0:
            values.append(dcg / ideal)
    return 0.0 if not values else float(np.mean(values))


def _weighted_dcg(
    examples: list[RankingExample],
    propensities: dict[int, float],
    *,
    k: int,
    min_propensity: float,
) -> float:
    return sum(
        _weighted_gain(example, propensities, min_propensity=min_propensity) / math.log2(rank + 1)
        for rank, example in enumerate(examples[:k], start=1)
    )


def _weighted_gain(
    example: RankingExample,
    propensities: dict[int, float],
    *,
    min_propensity: float,
) -> float:
    if example.position is None:
        raise ValueError("IPS evaluation requires every example to have an observed position")
    propensity = max(min_propensity, propensities.get(example.position, min_propensity))
    return ((2.0**example.label) - 1.0) / propensity
