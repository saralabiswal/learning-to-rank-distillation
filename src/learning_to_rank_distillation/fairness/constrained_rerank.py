"""Greedy constrained reranking."""

from __future__ import annotations

import math
from collections.abc import Sequence

from learning_to_rank_distillation.fairness.exposure import ExposureStats, low_exposure_groups
from learning_to_rank_distillation.schema import RankingExample, group_by_query


def constrained_rerank(
    examples: list[RankingExample],
    relevance_scores: Sequence[float],
    exposure_stats: dict[str, ExposureStats],
    *,
    exposure_floor: float,
) -> list[RankingExample]:
    """Greedily rerank one query list with a minimum low-exposure share.

    Implements FR-4.2.
    """

    if len(examples) != len(relevance_scores):
        raise ValueError("examples and relevance_scores must have the same length")
    if not 0.0 <= exposure_floor <= 1.0:
        raise ValueError("exposure_floor must be between 0 and 1")

    low_groups = low_exposure_groups(exposure_stats)
    candidates = list(zip(examples, relevance_scores, strict=True))
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    low_available = sum(1 for example, _ in candidates if example.group_id in low_groups)
    required_low = min(low_available, math.ceil(exposure_floor * len(candidates)))

    selected: list[RankingExample] = []
    selected_low = 0
    while candidates:
        remaining_slots = len(candidates)
        low_needed = required_low - selected_low
        must_pick_low = low_needed > 0 and remaining_slots == low_needed
        index = _best_low_exposure_index(candidates, low_groups) if must_pick_low else 0
        example, _ = candidates.pop(index)
        selected.append(example)
        if example.group_id in low_groups:
            selected_low += 1
    return selected


def rerank_by_query(
    examples: list[RankingExample],
    relevance_scores: Sequence[float],
    exposure_stats: dict[str, ExposureStats],
    *,
    exposure_floor: float,
) -> list[RankingExample]:
    """Apply constrained reranking independently within each query."""

    score_by_key = {
        (example.query_id, example.item_id): float(score)
        for example, score in zip(examples, relevance_scores, strict=True)
    }
    reranked: list[RankingExample] = []
    for query_examples in group_by_query(examples).values():
        query_scores = [
            score_by_key[(example.query_id, example.item_id)] for example in query_examples
        ]
        reranked.extend(
            constrained_rerank(
                query_examples,
                query_scores,
                exposure_stats,
                exposure_floor=exposure_floor,
            )
        )
    return reranked


def ranking_position_scores(examples: list[RankingExample]) -> list[float]:
    """Assign descending scores to an already ordered ranked output."""

    grouped = group_by_query(examples)
    scores_by_key: dict[tuple[str, str], float] = {}
    for query_examples in grouped.values():
        count = len(query_examples)
        for index, example in enumerate(query_examples):
            scores_by_key[(example.query_id, example.item_id)] = float(count - index)
    return [scores_by_key[(example.query_id, example.item_id)] for example in examples]


def _best_low_exposure_index(
    candidates: list[tuple[RankingExample, float]],
    low_groups: set[str],
) -> int:
    for index, (example, _) in enumerate(candidates):
        if example.group_id in low_groups:
            return index
    return 0
