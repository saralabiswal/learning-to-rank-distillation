from learning_to_rank_distillation.benchmark.metrics import ndcg_at_k
from learning_to_rank_distillation.schema import RankingExample


def test_ndcg_at_k_scores_perfect_ranking_as_one() -> None:
    examples = [
        RankingExample("q1", "i1", {"x": 1.0}, 2.0, "i1", True, 1),
        RankingExample("q1", "i2", {"x": 0.0}, 0.0, "i2", True, 2),
        RankingExample("q2", "i3", {"x": 1.0}, 1.0, "i3", True, 1),
        RankingExample("q2", "i4", {"x": 0.0}, 0.0, "i4", True, 2),
    ]

    assert ndcg_at_k(examples, [2.0, 0.0, 1.0, 0.0], k=2) == 1.0
