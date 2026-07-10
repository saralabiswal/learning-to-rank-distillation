from pathlib import Path

from learning_to_rank_distillation.benchmark.cross_dataset import run_cross_dataset_benchmark


def test_run_cross_dataset_benchmark_completes_available_and_skips_missing(
    tmp_path: Path,
) -> None:
    missing_rectour = tmp_path / "missing-rectour"

    results = run_cross_dataset_benchmark(
        datasets=("synthetic", "rectour"),
        data_dirs={"rectour": missing_rectour},
        output_dir=tmp_path / "cross",
        num_queries=8,
        items_per_query=4,
        student_epochs=1,
    )

    assert [(result.dataset, result.status) for result in results] == [
        ("synthetic", "completed"),
        ("rectour", "skipped"),
    ]
    assert (tmp_path / "cross" / "cross_dataset_benchmark.json").exists()
