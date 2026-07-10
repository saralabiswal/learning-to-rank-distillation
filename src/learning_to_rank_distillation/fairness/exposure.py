"""Supply exposure metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning_to_rank_distillation.schema import RankingExample


@dataclass(frozen=True, slots=True)
class ExposureStats:
    group_id: str
    impressions: int
    outcomes: int
    exposure_share: float


def compute_exposure_stats(examples: list[RankingExample]) -> dict[str, ExposureStats]:
    """Compute historical exposure and outcome counts from a training split.

    Implements DR-3 and FR-4.1.
    """

    if not examples:
        raise ValueError("examples must be non-empty")
    impression_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    for example in examples:
        impression_counts[example.group_id] = impression_counts.get(example.group_id, 0) + 1
        if example.label > 0:
            outcome_counts[example.group_id] = outcome_counts.get(example.group_id, 0) + 1

    total_impressions = sum(impression_counts.values())
    return {
        group_id: ExposureStats(
            group_id=group_id,
            impressions=impressions,
            outcomes=outcome_counts.get(group_id, 0),
            exposure_share=impressions / total_impressions,
        )
        for group_id, impressions in sorted(impression_counts.items())
    }


def low_exposure_groups(
    stats: dict[str, ExposureStats],
    *,
    quantile: float = 0.25,
) -> set[str]:
    """Return groups in the bottom exposure-share quantile."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    shares = np.asarray([value.exposure_share for value in stats.values()], dtype=np.float32)
    threshold = float(np.quantile(shares, quantile))
    return {group_id for group_id, value in stats.items() if value.exposure_share <= threshold}


def gini(values: list[float] | np.ndarray) -> float:
    """Compute the Gini coefficient for non-negative values."""

    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0):
        raise ValueError("gini values must be non-negative")
    if array.size == 0:
        raise ValueError("gini values must be non-empty")
    if np.all(array == 0):
        return 0.0
    sorted_array = np.sort(array)
    index = np.arange(1, sorted_array.size + 1)
    numerator = np.sum((2 * index - sorted_array.size - 1) * sorted_array)
    return float(numerator / (sorted_array.size * np.sum(sorted_array)))


def exposure_gini(examples: list[RankingExample], *, top_k: int | None = None) -> float:
    """Compute Gini over observed impressions in a ranked output."""

    counts: dict[str, int] = {}
    seen_by_query: dict[str, int] = {}
    for example in examples:
        rank = seen_by_query.get(example.query_id, 0) + 1
        seen_by_query[example.query_id] = rank
        if top_k is None or rank <= top_k:
            counts[example.group_id] = counts.get(example.group_id, 0) + 1
    return gini(list(counts.values()))


def low_exposure_impression_share(
    examples: list[RankingExample],
    low_exposure_group_ids: set[str],
    *,
    top_k: int = 5,
) -> float:
    """Share of top-k impressions allocated to historically low-exposure groups."""

    low_count = 0
    total = 0
    seen_by_query: dict[str, int] = {}
    for example in examples:
        rank = seen_by_query.get(example.query_id, 0) + 1
        seen_by_query[example.query_id] = rank
        if rank <= top_k:
            total += 1
            if example.group_id in low_exposure_group_ids:
                low_count += 1
    return 0.0 if total == 0 else low_count / total
