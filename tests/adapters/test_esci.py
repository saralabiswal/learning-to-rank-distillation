from pathlib import Path

import pytest

from learning_to_rank_distillation.adapters.esci import (
    ESCIAdapter,
    ESCIDataNotFoundError,
    ESCISchemaError,
)


def test_esci_adapter_reports_missing_data(tmp_path: Path) -> None:
    with pytest.raises(ESCIDataNotFoundError):
        ESCIAdapter(tmp_path / "missing").discover_files()


def test_esci_adapter_requires_examples_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "esci"
    data_dir.mkdir()
    (data_dir / "shopping_queries_dataset_products.csv").write_text(
        "product_id,product_locale,product_title\np1,us,boots\n",
        encoding="utf-8",
    )

    with pytest.raises(ESCIDataNotFoundError, match="shopping_queries_dataset_examples"):
        ESCIAdapter(data_dir).discover_files()


def test_esci_adapter_validates_required_columns(tmp_path: Path) -> None:
    data_dir = tmp_path / "esci"
    data_dir.mkdir()
    (data_dir / "shopping_queries_dataset_examples.csv").write_text(
        "query_id,product_id,esci_label\nq1,p1,E\n",
        encoding="utf-8",
    )

    with pytest.raises(ESCISchemaError, match="product_locale"):
        ESCIAdapter(data_dir).load()


def test_esci_adapter_loads_csv_examples_and_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "esci"
    _write_sample_esci_data(data_dir)

    examples = ESCIAdapter(data_dir).load()

    assert [example.query_id for example in examples] == ["q1", "q1", "q2"]
    assert [example.item_id for example in examples] == ["p1", "p2", "p3"]
    assert [example.label for example in examples] == [3.0, 0.0, 2.0]
    assert examples[0].group_id == "Acme"
    assert examples[1].group_id == "p2"
    assert examples[0].is_unbiased is False
    assert examples[0].position is None
    assert examples[0].features["product_locale"] == "us"
    assert examples[0].features["source"] == "search"
    assert examples[0].features["query_title_token_overlap"] > 0
    assert examples[0].features["has_product_brand"] is True
    assert examples[1].features["product_brand"] is None
    assert "position" not in examples[0].features


def test_esci_adapter_applies_limit_after_filters(tmp_path: Path) -> None:
    data_dir = tmp_path / "esci"
    _write_sample_esci_data(data_dir)

    examples = ESCIAdapter(data_dir).load(limit=2)

    assert [example.item_id for example in examples] == ["p1", "p2"]


def _write_sample_esci_data(data_dir: Path) -> None:
    data_dir.mkdir()
    (data_dir / "shopping_queries_dataset_examples.csv").write_text(
        "example_id,query,query_id,product_id,product_locale,esci_label,small_version,split\n"
        "e1,waterproof hiking boots,q1,p1,us,E,1,train\n"
        "e2,waterproof hiking boots,q1,p2,us,I,1,train\n"
        "e3,linen shirt,q2,p3,us,S,1,train\n"
        "e4,linen shirt,q2,p4,us,C,0,train\n"
        "e5,trial product,q3,p5,us,C,1,test\n",
        encoding="utf-8",
    )
    (data_dir / "shopping_queries_dataset_products.csv").write_text(
        "product_id,product_locale,product_title,product_description,product_bullet_point,"
        "product_brand,product_color\n"
        "p1,us,Waterproof Hiking Boots,Durable boots for wet trails,"
        "waterproof leather hiking,Acme,Brown\n"
        "p2,us,Cotton Socks,Soft socks,ankle cotton,,White\n"
        "p3,us,Linen Shirt,Light linen shirt,summer linen,Northstar,Blue\n",
        encoding="utf-8",
    )
    (data_dir / "shopping_queries_dataset_sources.csv").write_text(
        "query_id,source\nq1,search\nq2,browse\n",
        encoding="utf-8",
    )
