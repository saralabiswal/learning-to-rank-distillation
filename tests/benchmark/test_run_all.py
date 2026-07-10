from pathlib import Path

from learning_to_rank_distillation.benchmark.run_all import run_benchmark


def test_run_benchmark_produces_comparison_table_and_plot(tmp_path: Path) -> None:
    rows = run_benchmark(
        num_queries=12,
        items_per_query=4,
        student_embedding_dims=(4, 8, 12),
        student_epochs=1,
        output_dir=tmp_path,
    )

    model_names = {row.model for row in rows}
    assert "teacher-lightgbm" in model_names
    assert "student-no-kd-d16" in model_names
    assert {"student-kd-d4", "student-kd-d8", "student-kd-d12"}.issubset(model_names)
    assert (tmp_path / "benchmark_table.json").exists()
    assert (tmp_path / "quality_latency_pareto.png").exists()
    assert (tmp_path / "fairness_tradeoff.png").exists()
    assert (tmp_path / "fairness_pareto_frontier.png").exists()
    assert (tmp_path / "promotion_registry.sqlite").exists()


def test_run_benchmark_supports_esci_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "esci"
    _write_esci_benchmark_data(data_dir)
    output_dir = tmp_path / "artifacts"

    rows = run_benchmark(
        dataset="esci",
        data_dir=data_dir,
        student_embedding_dims=(4,),
        student_epochs=1,
        output_dir=output_dir,
    )

    model_names = {row.model for row in rows}
    assert {"teacher-lightgbm", "student-no-kd-d16", "student-kd-d4"} == model_names
    assert (output_dir / "benchmark_table.json").exists()
    assert (output_dir / "fairness_tradeoff.json").exists()
    assert (output_dir / "fairness_pareto_frontier.json").exists()


def _write_esci_benchmark_data(data_dir: Path) -> None:
    data_dir.mkdir()
    example_lines = [
        "example_id,query,query_id,product_id,product_locale,esci_label,small_version,split"
    ]
    product_lines = [
        "product_id,product_locale,product_title,product_description,product_bullet_point,"
        "product_brand,product_color"
    ]
    source_lines = ["query_id,source"]
    labels = ("E", "S", "I")
    for query_index in range(8):
        query_id = f"q{query_index}"
        term = "boots" if query_index % 2 == 0 else "shirt"
        source_lines.append(f"{query_id},search")
        for item_index, label in enumerate(labels):
            product_id = f"p{query_index}-{item_index}"
            brand = f"Brand{item_index % 2}"
            color = "Brown" if item_index == 0 else "Blue"
            title = f"{term} product {item_index}"
            example_lines.append(
                f"e{query_index}-{item_index},{term},{query_id},{product_id},us,{label},1,train"
            )
            product_lines.append(
                f"{product_id},us,{title},Useful {term} item,{term} durable,{brand},{color}"
            )
    (data_dir / "shopping_queries_dataset_examples.csv").write_text(
        "\n".join(example_lines) + "\n",
        encoding="utf-8",
    )
    (data_dir / "shopping_queries_dataset_products.csv").write_text(
        "\n".join(product_lines) + "\n",
        encoding="utf-8",
    )
    (data_dir / "shopping_queries_dataset_sources.csv").write_text(
        "\n".join(source_lines) + "\n",
        encoding="utf-8",
    )
