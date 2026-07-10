from pathlib import Path

from learning_to_rank_distillation.adapters.synthetic import make_synthetic_ranking_data
from learning_to_rank_distillation.benchmark.fairness_tradeoff import run_fairness_tradeoff
from learning_to_rank_distillation.data import split_by_query


def test_run_fairness_tradeoff_writes_outputs(tmp_path: Path) -> None:
    examples = make_synthetic_ranking_data(num_queries=8, items_per_query=4, seed=9)
    split = split_by_query(examples, seed=9)
    relevance_scores = [example.label for example in split.test]

    rows = run_fairness_tradeoff(
        train_examples=split.train,
        eval_examples=split.test,
        relevance_scores=relevance_scores,
        exposure_floors=(0.0, 0.25),
        output_dir=tmp_path,
    )

    assert len(rows) == 2
    assert (tmp_path / "fairness_tradeoff.json").exists()
    assert (tmp_path / "fairness_tradeoff.png").exists()
