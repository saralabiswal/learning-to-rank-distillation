import sqlite3
from pathlib import Path

from learning_to_rank_distillation.governance.promotion_gate import ModelMetrics, PromotionGate


def test_promotion_gate_logs_decision(tmp_path: Path) -> None:
    registry_path = tmp_path / "promotion.sqlite"
    gate = PromotionGate(registry_path=registry_path)

    decision = gate.evaluate(
        candidate=ModelMetrics(
            model_version="student-kd-d16",
            ndcg_at_5=0.99,
            ndcg_at_10=0.99,
            latency_p99_ms=10.0,
            data_hash="abc",
        ),
        teacher=ModelMetrics(
            model_version="teacher-lightgbm",
            ndcg_at_5=1.0,
            ndcg_at_10=1.0,
            latency_p99_ms=40.0,
            data_hash="abc",
        ),
    )

    with sqlite3.connect(registry_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM promotion_decisions").fetchone()[0]

    assert decision.promoted
    assert len(decision.stages) == 8
    assert count == 1
