from pathlib import Path

from learning_to_rank_distillation.benchmark.distillation_ablation import (
    run_distillation_ablation,
)


def test_run_distillation_ablation_writes_method_comparison(tmp_path: Path) -> None:
    rows = run_distillation_ablation(
        num_queries=8,
        items_per_query=4,
        output_dir=tmp_path,
        student_epochs=1,
        teacher_epochs=1,
        embedding_dim=8,
    )

    assert {row.model for row in rows} == {
        "teacher-transformer",
        "student-no-kd-d8",
        "student-response-kd-d8",
        "student-feature-kd-d8",
        "student-relation-kd-d8",
    }
    assert (tmp_path / "distillation_ablation.json").exists()
