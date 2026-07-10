import pytest

from learning_to_rank_distillation.evaluation.ips import (
    estimate_position_propensities,
    ips_ndcg_at_k,
)
from learning_to_rank_distillation.schema import RankingExample


def test_estimate_position_propensities_counts_observed_positions() -> None:
    examples = [
        _example("q1", "i1", label=1.0, position=1),
        _example("q1", "i2", label=0.0, position=2),
        _example("q2", "i3", label=1.0, position=1),
    ]

    propensities = estimate_position_propensities(examples)

    assert propensities == {1: 1.0, 2: 0.5}


def test_ips_ndcg_at_k_scores_perfect_weighted_ranking_as_one() -> None:
    examples = [
        _example("q1", "i1", label=2.0, position=2),
        _example("q1", "i2", label=0.0, position=1),
        _example("q1", "i3", label=1.0, position=3),
    ]

    score = ips_ndcg_at_k(
        examples,
        [3.0, 1.0, 2.0],
        k=3,
        propensities={1: 1.0, 2: 0.5, 3: 0.25},
    )

    assert score == pytest.approx(1.0)


def test_ips_ndcg_at_k_requires_positions() -> None:
    examples = [_example("q1", "i1", label=1.0, position=None)]

    with pytest.raises(ValueError, match="observed position"):
        ips_ndcg_at_k(examples, [1.0], k=1)


def _example(
    query_id: str,
    item_id: str,
    *,
    label: float,
    position: int | None,
) -> RankingExample:
    return RankingExample(
        query_id=query_id,
        item_id=item_id,
        group_id=item_id,
        label=label,
        is_unbiased=position is not None,
        position=position,
        features={"x": 1.0},
    )
