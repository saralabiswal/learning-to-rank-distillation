"""Pareto-search utilities for relevance vs. exposure fairness."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from learning_to_rank_distillation.benchmark.metrics import ndcg_at_k
from learning_to_rank_distillation.fairness.constrained_rerank import ranking_position_scores
from learning_to_rank_distillation.fairness.exposure import (
    ExposureStats,
    compute_exposure_stats,
    exposure_gini,
    low_exposure_groups,
    low_exposure_impression_share,
)
from learning_to_rank_distillation.schema import RankingExample, group_by_query


@dataclass(frozen=True, slots=True)
class ParetoSearchRow:
    fairness_weight: float
    ndcg_at_5: float
    low_exposure_impression_share: float
    exposure_gini_at_5: float
    is_pareto_efficient: bool


def scalarized_rerank_by_query(
    examples: list[RankingExample],
    relevance_scores: Sequence[float],
    exposure_stats: dict[str, ExposureStats],
    *,
    fairness_weight: float,
) -> list[RankingExample]:
    """Rerank each query by relevance plus a continuous exposure-fairness bonus."""

    if len(examples) != len(relevance_scores):
        raise ValueError("examples and relevance_scores must have the same length")
    if fairness_weight < 0:
        raise ValueError("fairness_weight must be non-negative")

    score_by_key = {
        (example.query_id, example.item_id): float(score)
        for example, score in zip(examples, relevance_scores, strict=True)
    }
    reranked: list[RankingExample] = []
    for query_examples in group_by_query(examples).values():
        query_scores = np.asarray(
            [score_by_key[(example.query_id, example.item_id)] for example in query_examples],
            dtype=np.float32,
        )
        relevance_component = _standardize(query_scores)
        combined = [
            (
                example,
                float(relevance_component[index])
                + fairness_weight * _exposure_bonus(example.group_id, exposure_stats),
            )
            for index, example in enumerate(query_examples)
        ]
        combined.sort(key=lambda pair: pair[1], reverse=True)
        reranked.extend(example for example, _ in combined)
    return reranked


def pareto_frontier_search(
    *,
    train_examples: list[RankingExample],
    eval_examples: list[RankingExample],
    relevance_scores: Sequence[float],
    fairness_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    top_k: int = 5,
) -> list[ParetoSearchRow]:
    """Run a scalarization sweep and mark non-dominated relevance/fairness points."""

    exposure_stats = compute_exposure_stats(train_examples)
    low_groups = low_exposure_groups(exposure_stats)
    rows: list[ParetoSearchRow] = []
    for weight in fairness_weights:
        reranked = scalarized_rerank_by_query(
            eval_examples,
            relevance_scores,
            exposure_stats,
            fairness_weight=weight,
        )
        position_scores = ranking_position_scores(reranked)
        rows.append(
            ParetoSearchRow(
                fairness_weight=weight,
                ndcg_at_5=ndcg_at_k(reranked, position_scores, k=top_k),
                low_exposure_impression_share=low_exposure_impression_share(
                    reranked,
                    low_groups,
                    top_k=top_k,
                ),
                exposure_gini_at_5=exposure_gini(reranked, top_k=top_k),
                is_pareto_efficient=False,
            )
        )

    efficient = _pareto_efficient_mask(rows)
    return [
        ParetoSearchRow(
            fairness_weight=row.fairness_weight,
            ndcg_at_5=row.ndcg_at_5,
            low_exposure_impression_share=row.low_exposure_impression_share,
            exposure_gini_at_5=row.exposure_gini_at_5,
            is_pareto_efficient=is_efficient,
        )
        for row, is_efficient in zip(rows, efficient, strict=True)
    ]


def _exposure_bonus(group_id: str, exposure_stats: dict[str, ExposureStats]) -> float:
    if group_id not in exposure_stats:
        return 1.0
    shares = np.asarray([stats.exposure_share for stats in exposure_stats.values()])
    min_share = float(shares.min())
    max_share = float(shares.max())
    if max_share == min_share:
        return 0.0
    exposure_share = exposure_stats[group_id].exposure_share
    return 1.0 - ((exposure_share - min_share) / (max_share - min_share))


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std == 0.0:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _pareto_efficient_mask(rows: list[ParetoSearchRow]) -> list[bool]:
    objectives = np.asarray(
        [
            [row.ndcg_at_5, row.low_exposure_impression_share, -row.exposure_gini_at_5]
            for row in rows
        ],
        dtype=np.float32,
    )
    efficient: list[bool] = []
    for index, objective in enumerate(objectives):
        others = np.delete(objectives, index, axis=0)
        dominated = bool(
            np.any(np.all(others >= objective, axis=1) & np.any(others > objective, axis=1))
        )
        efficient.append(not dominated)
    return efficient
