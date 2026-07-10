"""Dataset utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass

from learning_to_rank_distillation.schema import RankingExample, group_by_query


@dataclass(frozen=True, slots=True)
class QuerySplit:
    train: list[RankingExample]
    validation: list[RankingExample]
    test: list[RankingExample]


def split_by_query(
    examples: list[RankingExample],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 13,
) -> QuerySplit:
    """Split examples by query_id, never by row.

    Implements DR-2.
    """

    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("fractions must be non-negative")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    grouped = group_by_query(examples)
    query_ids = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(query_ids)

    total = len(query_ids)
    test_count = max(1, round(total * test_fraction))
    validation_count = max(1, round(total * validation_fraction))
    if validation_count + test_count >= total:
        raise ValueError("not enough query groups for requested split fractions")

    test_ids = set(query_ids[:test_count])
    validation_ids = set(query_ids[test_count : test_count + validation_count])
    train_ids = set(query_ids[test_count + validation_count :])

    return QuerySplit(
        train=_flatten_groups(grouped, train_ids),
        validation=_flatten_groups(grouped, validation_ids),
        test=_flatten_groups(grouped, test_ids),
    )


def _flatten_groups(
    grouped: dict[str, list[RankingExample]], query_ids: set[str]
) -> list[RankingExample]:
    return [
        example
        for query_id, query_examples in grouped.items()
        if query_id in query_ids
        for example in query_examples
    ]
