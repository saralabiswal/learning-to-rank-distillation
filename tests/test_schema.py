import pytest

from learning_to_rank_distillation.schema import RankingExample, group_by_query, validate_examples


def test_ranking_example_rejects_position_feature() -> None:
    with pytest.raises(ValueError, match="position"):
        RankingExample(
            query_id="q1",
            item_id="i1",
            group_id="i1",
            label=1.0,
            is_unbiased=True,
            position=1,
            features={"position": 1},
        )


def test_group_by_query_preserves_examples() -> None:
    examples = [
        RankingExample("q1", "i1", {"score": 1.0}, 1.0, "i1", True, 1),
        RankingExample("q1", "i2", {"score": 0.0}, 0.0, "i2", True, 2),
        RankingExample("q2", "i3", {"score": 0.5}, 1.0, "i3", False, None),
    ]

    validate_examples(examples)
    grouped = group_by_query(examples)

    assert list(grouped) == ["q1", "q2"]
    assert [example.item_id for example in grouped["q1"]] == ["i1", "i2"]
