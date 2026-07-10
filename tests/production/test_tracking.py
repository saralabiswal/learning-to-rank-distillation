import json
from dataclasses import dataclass
from pathlib import Path

from learning_to_rank_distillation.production.tracking import (
    benchmark_metrics_from_rows,
    log_experiment_run,
)


@dataclass(frozen=True)
class _Row:
    model: str
    ndcg_at_5: float
    latency_p99_ms: float


def test_log_experiment_run_writes_jsonl(tmp_path: Path) -> None:
    tracking_path = log_experiment_run(
        run_name="test-run",
        metrics={"ndcg": 0.5},
        params={"dataset": "synthetic"},
        tracking_path=tmp_path / "experiments.jsonl",
    )

    record = json.loads(tracking_path.read_text(encoding="utf-8"))

    assert record["run_name"] == "test-run"
    assert record["metrics"] == {"ndcg": 0.5}
    assert record["params"] == {"dataset": "synthetic"}


def test_benchmark_metrics_from_rows_flattens_model_names() -> None:
    metrics = benchmark_metrics_from_rows([_Row("student-kd-d8", 0.7, 1.2)])

    assert metrics == {
        "student_kd_d8.ndcg_at_5": 0.7,
        "student_kd_d8.latency_p99_ms": 1.2,
    }
