"""Governed model promotion gate."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    model_version: str
    ndcg_at_5: float
    ndcg_at_10: float
    latency_p99_ms: float
    data_hash: str


@dataclass(frozen=True, slots=True)
class StageResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    candidate_model_version: str
    teacher_model_version: str
    promoted: bool
    ndcg_drop: float
    latency_improvement: float
    stages: list[StageResult]
    timestamp: str


@dataclass(slots=True)
class PromotionGate:
    """Eight-stage promotion gate with local SQLite logging.

    Implements FR-5.1, FR-5.2, and FR-5.3.
    """

    registry_path: Path = Path("artifacts/promotion_registry.sqlite")
    max_ndcg_drop: float = 0.02
    min_latency_improvement: float = 3.0

    def evaluate(
        self,
        *,
        candidate: ModelMetrics,
        teacher: ModelMetrics,
    ) -> PromotionDecision:
        ndcg_drop = _relative_drop(teacher.ndcg_at_5, candidate.ndcg_at_5)
        latency_improvement = teacher.latency_p99_ms / candidate.latency_p99_ms
        stages = [
            StageResult(
                "candidate_registered",
                bool(candidate.model_version),
                "candidate model version is present",
            ),
            StageResult(
                "teacher_registered",
                bool(teacher.model_version),
                "teacher model version is present",
            ),
            StageResult(
                "quality_metric_present",
                0.0 <= candidate.ndcg_at_5 <= 1.0 and 0.0 <= teacher.ndcg_at_5 <= 1.0,
                "NDCG@5 values are bounded",
            ),
            StageResult(
                "secondary_quality_metric_present",
                0.0 <= candidate.ndcg_at_10 <= 1.0 and 0.0 <= teacher.ndcg_at_10 <= 1.0,
                "NDCG@10 values are bounded",
            ),
            StageResult(
                "latency_metric_present",
                candidate.latency_p99_ms > 0.0 and teacher.latency_p99_ms > 0.0,
                "p99 latency values are positive",
            ),
            StageResult(
                "lineage_hash_present",
                bool(candidate.data_hash) and bool(teacher.data_hash),
                "training data hashes are present",
            ),
            StageResult(
                "quality_gate",
                ndcg_drop <= self.max_ndcg_drop,
                f"NDCG@5 relative drop {ndcg_drop:.4f} <= {self.max_ndcg_drop:.4f}",
            ),
            StageResult(
                "latency_gate",
                latency_improvement >= self.min_latency_improvement,
                "p99 latency improvement "
                f"{latency_improvement:.4f} >= {self.min_latency_improvement:.4f}",
            ),
        ]
        decision = PromotionDecision(
            candidate_model_version=candidate.model_version,
            teacher_model_version=teacher.model_version,
            promoted=all(stage.passed for stage in stages),
            ndcg_drop=ndcg_drop,
            latency_improvement=latency_improvement,
            stages=stages,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.log_decision(decision, candidate=candidate, teacher=teacher)
        return decision

    def log_decision(
        self,
        decision: PromotionDecision,
        *,
        candidate: ModelMetrics,
        teacher: ModelMetrics,
    ) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.registry_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    candidate_model_version TEXT NOT NULL,
                    teacher_model_version TEXT NOT NULL,
                    promoted INTEGER NOT NULL,
                    ndcg_drop REAL NOT NULL,
                    latency_improvement REAL NOT NULL,
                    candidate_metrics TEXT NOT NULL,
                    teacher_metrics TEXT NOT NULL,
                    stages TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO promotion_decisions (
                    timestamp,
                    candidate_model_version,
                    teacher_model_version,
                    promoted,
                    ndcg_drop,
                    latency_improvement,
                    candidate_metrics,
                    teacher_metrics,
                    stages
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.timestamp,
                    decision.candidate_model_version,
                    decision.teacher_model_version,
                    int(decision.promoted),
                    decision.ndcg_drop,
                    decision.latency_improvement,
                    json.dumps(asdict(candidate), sort_keys=True),
                    json.dumps(asdict(teacher), sort_keys=True),
                    json.dumps([asdict(stage) for stage in decision.stages], sort_keys=True),
                ),
            )


def _relative_drop(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate >= baseline else 1.0
    return max(0.0, (baseline - candidate) / baseline)
