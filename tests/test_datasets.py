from pathlib import Path

from learning_to_rank_distillation.datasets import DatasetConfig, load_ranking_examples


def test_load_ranking_examples_uses_synthetic_defaults() -> None:
    examples = load_ranking_examples(
        DatasetConfig(name="synthetic", num_queries=3, items_per_query=2, seed=7)
    )

    assert len(examples) == 6
    assert {example.query_id for example in examples} == {"q-00000", "q-00001", "q-00002"}


def test_load_ranking_examples_uses_esci_adapter(tmp_path: Path) -> None:
    data_dir = tmp_path / "esci"
    data_dir.mkdir()
    (data_dir / "shopping_queries_dataset_examples.csv").write_text(
        "query_id,product_id,product_locale,esci_label,small_version,split\n"
        "q1,p1,us,E,1,train\n"
        "q1,p2,us,I,1,train\n",
        encoding="utf-8",
    )

    examples = load_ranking_examples(DatasetConfig(name="esci", data_dir=data_dir))

    assert [example.label for example in examples] == [3.0, 0.0]


def test_load_ranking_examples_uses_movielens_adapter(tmp_path: Path) -> None:
    data_dir = tmp_path / "movielens"
    data_dir.mkdir()
    (data_dir / "ratings.csv").write_text(
        "userId,movieId,rating\n1,10,4.0\n1,20,3.0\n",
        encoding="utf-8",
    )

    examples = load_ranking_examples(DatasetConfig(name="movielens", data_dir=data_dir))

    assert [example.query_id for example in examples] == ["user-1", "user-1"]
    assert [example.label for example in examples] == [4.0, 3.0]
