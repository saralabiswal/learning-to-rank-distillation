from pathlib import Path

from learning_to_rank_distillation.adapters.rectour import RecTourAdapter
from learning_to_rank_distillation.adapters.synthetic import (
    RECTOUR_LIKE_FEATURE_COLUMNS,
    make_synthetic_rectour_rows,
    write_synthetic_rectour_csv,
)
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_synthetic_data_matches_schema() -> None:
    examples = make_synthetic_ranking_data(num_queries=4, items_per_query=3, seed=7)

    assert len(examples) == 12
    assert len({example.query_id for example in examples}) == 4
    assert all("position" not in example.features for example in examples)
    assert {example.label for example in examples}.issubset({0.0, 1.0, 2.0})
    assert set(RECTOUR_LIKE_FEATURE_COLUMNS).issubset(examples[0].features)


def test_synthetic_rectour_rows_expose_documented_shape() -> None:
    rows = make_synthetic_rectour_rows(num_queries=4, items_per_query=5, seed=11)

    assert len(rows) == 20
    assert {
        "query_id",
        "item_id",
        "group_id",
        "prop_id",
        "destination_id",
        "num_clicks",
        "is_trans",
        "label",
        "is_unbiased",
        "position",
        "review_rating",
        "review_count",
        "price_bucket",
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "price_usd",
        "prop_location_score1",
        "prop_location_score2",
        "promotion_flag",
        "competitor_price_delta",
        "competitor_availability_rate",
        "price_rank",
        "star_rank",
        "location_score_rank",
        "random_bool",
    }.issubset(rows[0])
    assert {row["label"] for row in rows}.issubset({0.0, 1.0, 2.0})
    assert any(row["is_unbiased"] for row in rows)
    assert any(not row["is_unbiased"] for row in rows)
    assert "random_bool" not in RECTOUR_LIKE_FEATURE_COLUMNS


def test_synthetic_rectour_csv_round_trips_through_adapter(tmp_path: Path) -> None:
    csv_path = write_synthetic_rectour_csv(
        tmp_path / "rectour_like.csv",
        num_queries=3,
        items_per_query=4,
        seed=19,
    )

    examples = RecTourAdapter(tmp_path).load()

    assert csv_path.exists()
    assert len(examples) == 12
    assert all("position" not in example.features for example in examples)
    assert any(example.is_unbiased for example in examples)
