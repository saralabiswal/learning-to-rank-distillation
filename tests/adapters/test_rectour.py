from pathlib import Path

import pytest

from learning_to_rank_distillation.adapters.rectour import (
    RecTourAdapter,
    RecTourColumnMapping,
    RecTourDataNotFoundError,
    RecTourSchemaError,
)


def test_rectour_adapter_reports_missing_data(tmp_path: Path) -> None:
    with pytest.raises(RecTourDataNotFoundError):
        RecTourAdapter(tmp_path / "missing").discover_files()


def test_rectour_adapter_requires_explicit_mapping_for_unknown_schema(tmp_path: Path) -> None:
    data_dir = tmp_path / "rectour"
    data_dir.mkdir()
    (data_dir / "sample.csv").write_text("search,property,clicked\nq1,p1,1\n", encoding="utf-8")

    with pytest.raises(RecTourSchemaError, match="explicit RecTourColumnMapping"):
        RecTourAdapter(data_dir).load()


def test_rectour_adapter_loads_explicit_mapping(tmp_path: Path) -> None:
    data_dir = tmp_path / "rectour"
    data_dir.mkdir()
    (data_dir / "sample.csv").write_text(
        "search,property,clicked,rank_randomized,observed_position,review_rating\n"
        "q1,p1,1,true,1,4.5\n"
        "q1,p2,0,true,2,3.5\n",
        encoding="utf-8",
    )

    examples = RecTourAdapter(
        data_dir,
        RecTourColumnMapping(
            query_id="search",
            item_id="property",
            label="clicked",
            is_unbiased="rank_randomized",
            position="observed_position",
            feature_columns=("review_rating",),
        ),
    ).load()

    assert [example.item_id for example in examples] == ["p1", "p2"]
    assert examples[0].group_id == "p1"
    assert examples[0].position == 1
    assert examples[0].features == {"review_rating": 4.5}
