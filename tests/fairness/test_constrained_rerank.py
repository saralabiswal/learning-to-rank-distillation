from learning_to_rank_distillation.fairness.constrained_rerank import constrained_rerank
from learning_to_rank_distillation.fairness.exposure import compute_exposure_stats
from learning_to_rank_distillation.schema import RankingExample


def test_constrained_rerank_reserves_low_exposure_slots() -> None:
    train = [
        RankingExample("q0", "a", {"x": 1.0}, 1.0, "high", True, 1),
        RankingExample("q1", "b", {"x": 1.0}, 1.0, "high", True, 1),
        RankingExample("q2", "c", {"x": 1.0}, 1.0, "low", True, 1),
    ]
    candidates = [
        RankingExample("q3", "a", {"x": 1.0}, 1.0, "high", True, 1),
        RankingExample("q3", "b", {"x": 1.0}, 1.0, "high", True, 2),
        RankingExample("q3", "c", {"x": 1.0}, 1.0, "low", True, 3),
    ]

    reranked = constrained_rerank(
        candidates,
        [3.0, 2.0, 1.0],
        compute_exposure_stats(train),
        exposure_floor=0.34,
    )

    assert reranked[-1].group_id == "low"
    assert {example.item_id for example in reranked} == {"a", "b", "c"}
