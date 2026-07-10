from learning_to_rank_distillation.fairness.exposure import compute_exposure_stats
from learning_to_rank_distillation.fairness.pareto import (
    pareto_frontier_search,
    scalarized_rerank_by_query,
)
from learning_to_rank_distillation.schema import RankingExample


def test_scalarized_rerank_can_promote_low_exposure_group() -> None:
    train_examples = [
        _example("train-1", f"popular-{index}", "popular", 0.0) for index in range(5)
    ] + [_example("train-2", "rare-1", "rare", 0.0)]
    eval_examples = [
        _example("eval-1", "popular-candidate", "popular", 0.0),
        _example("eval-1", "rare-candidate", "rare", 0.0),
    ]
    stats = compute_exposure_stats(train_examples)

    relevance_first = scalarized_rerank_by_query(
        eval_examples,
        [2.0, 1.0],
        stats,
        fairness_weight=0.0,
    )
    fairness_first = scalarized_rerank_by_query(
        eval_examples,
        [2.0, 1.0],
        stats,
        fairness_weight=4.0,
    )

    assert relevance_first[0].group_id == "popular"
    assert fairness_first[0].group_id == "rare"


def test_pareto_frontier_search_marks_efficient_rows() -> None:
    train_examples = [
        _example("train-1", f"popular-{index}", "popular", 0.0) for index in range(5)
    ] + [_example("train-2", "rare-1", "rare", 0.0)]
    eval_examples = [
        _example("eval-1", "popular-candidate", "popular", 2.0),
        _example("eval-1", "rare-candidate", "rare", 1.0),
    ]

    rows = pareto_frontier_search(
        train_examples=train_examples,
        eval_examples=eval_examples,
        relevance_scores=[2.0, 1.0],
        fairness_weights=(0.0, 4.0),
    )

    assert len(rows) == 2
    assert any(row.is_pareto_efficient for row in rows)


def _example(query_id: str, item_id: str, group_id: str, label: float) -> RankingExample:
    return RankingExample(
        query_id=query_id,
        item_id=item_id,
        group_id=group_id,
        label=label,
        is_unbiased=True,
        position=None,
        features={"bias": 1.0},
    )
