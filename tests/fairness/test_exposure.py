from learning_to_rank_distillation.fairness.exposure import (
    compute_exposure_stats,
    gini,
    low_exposure_groups,
)
from learning_to_rank_distillation.schema import RankingExample


def test_compute_exposure_stats_counts_impressions_and_outcomes() -> None:
    examples = [
        RankingExample("q1", "i1", {"x": 1.0}, 1.0, "g1", True, 1),
        RankingExample("q2", "i1", {"x": 1.0}, 0.0, "g1", True, 1),
        RankingExample("q3", "i2", {"x": 1.0}, 2.0, "g2", True, 1),
    ]

    stats = compute_exposure_stats(examples)

    assert stats["g1"].impressions == 2
    assert stats["g1"].outcomes == 1
    assert stats["g2"].impressions == 1
    assert low_exposure_groups(stats) == {"g2"}


def test_gini_handles_equal_distribution() -> None:
    assert gini([1, 1, 1]) == 0.0
