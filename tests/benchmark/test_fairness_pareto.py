from pathlib import Path

from learning_to_rank_distillation.benchmark.fairness_pareto import run_fairness_pareto_search
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_run_fairness_pareto_search_writes_outputs(tmp_path: Path) -> None:
    examples = make_synthetic_ranking_data(num_queries=6, items_per_query=4, seed=31)
    train_examples = examples[:16]
    eval_examples = examples[16:]
    relevance_scores = [float(example.label) for example in eval_examples]

    rows = run_fairness_pareto_search(
        train_examples=train_examples,
        eval_examples=eval_examples,
        relevance_scores=relevance_scores,
        output_dir=tmp_path,
    )

    assert rows
    assert (tmp_path / "fairness_pareto_frontier.json").exists()
    assert (tmp_path / "fairness_pareto_frontier.png").exists()
