import json
from pathlib import Path

from learning_to_rank_distillation.benchmark.promotion_check import check_benchmark_promotion


def test_check_benchmark_promotion_promotes_best_candidate(tmp_path: Path) -> None:
    table_path = tmp_path / "benchmark_table.json"
    _write_table(
        table_path,
        teacher_ndcg=0.8,
        candidate_ndcg=0.79,
        teacher_p99=9.0,
        candidate_p99=2.0,
    )

    decision = check_benchmark_promotion(
        benchmark_table=table_path,
        registry_path=tmp_path / "registry.sqlite",
    )

    assert decision.promoted is True


def test_check_benchmark_promotion_fails_quality_regression(tmp_path: Path) -> None:
    table_path = tmp_path / "benchmark_table.json"
    _write_table(
        table_path,
        teacher_ndcg=0.8,
        candidate_ndcg=0.6,
        teacher_p99=9.0,
        candidate_p99=2.0,
    )

    decision = check_benchmark_promotion(
        benchmark_table=table_path,
        registry_path=tmp_path / "registry.sqlite",
    )

    assert decision.promoted is False
    assert any(stage.name == "quality_gate" and not stage.passed for stage in decision.stages)


def _write_table(
    path: Path,
    *,
    teacher_ndcg: float,
    candidate_ndcg: float,
    teacher_p99: float,
    candidate_p99: float,
) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "model": "teacher-lightgbm",
                    "ndcg_at_5": teacher_ndcg,
                    "ndcg_at_10": teacher_ndcg,
                    "latency_p99_ms": teacher_p99,
                },
                {
                    "model": "student-kd-d16",
                    "ndcg_at_5": candidate_ndcg,
                    "ndcg_at_10": candidate_ndcg,
                    "latency_p99_ms": candidate_p99,
                },
            ]
        ),
        encoding="utf-8",
    )
