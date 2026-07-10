"""CLI check for CI-enforced benchmark promotion gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from learning_to_rank_distillation.governance.promotion_gate import (
    ModelMetrics,
    PromotionDecision,
    PromotionGate,
)


def check_benchmark_promotion(
    *,
    benchmark_table: Path,
    registry_path: Path,
    max_ndcg_drop: float = 0.02,
    min_latency_improvement: float = 3.0,
    candidate_prefix: str = "student-kd",
) -> PromotionDecision:
    """Evaluate the best candidate in a benchmark table against the teacher row."""

    rows = json.loads(benchmark_table.read_text(encoding="utf-8"))
    teacher_row = _find_teacher(rows)
    candidate_row = _find_best_candidate(rows, candidate_prefix=candidate_prefix)
    data_hash = _file_hash(benchmark_table)
    decision = PromotionGate(
        registry_path=registry_path,
        max_ndcg_drop=max_ndcg_drop,
        min_latency_improvement=min_latency_improvement,
    ).evaluate(
        candidate=_metrics(candidate_row, data_hash=data_hash),
        teacher=_metrics(teacher_row, data_hash=data_hash),
    )
    return decision


def _find_teacher(rows: list[dict[str, object]]) -> dict[str, object]:
    for row in rows:
        if row.get("model") == "teacher-lightgbm":
            return row
    raise ValueError("benchmark table has no teacher-lightgbm row")


def _find_best_candidate(
    rows: list[dict[str, object]],
    *,
    candidate_prefix: str,
) -> dict[str, object]:
    candidates = [row for row in rows if str(row.get("model", "")).startswith(candidate_prefix)]
    if not candidates:
        raise ValueError(f"benchmark table has no candidate rows matching {candidate_prefix!r}")
    return max(candidates, key=lambda row: float(row["ndcg_at_5"]))


def _metrics(row: dict[str, object], *, data_hash: str) -> ModelMetrics:
    return ModelMetrics(
        model_version=str(row["model"]),
        ndcg_at_5=float(row["ndcg_at_5"]),
        ndcg_at_10=float(row["ndcg_at_10"]),
        latency_p99_ms=float(row["latency_p99_ms"]),
        data_hash=data_hash,
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fail if benchmark promotion policy fails.")
    parser.add_argument("--benchmark-table", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, default=Path("artifacts/promotion_ci.sqlite"))
    parser.add_argument("--max-ndcg-drop", type=float, default=0.02)
    parser.add_argument("--min-latency-improvement", type=float, default=3.0)
    parser.add_argument("--candidate-prefix", default="student-kd")
    args = parser.parse_args(argv)

    decision = check_benchmark_promotion(
        benchmark_table=args.benchmark_table,
        registry_path=args.registry_path,
        max_ndcg_drop=args.max_ndcg_drop,
        min_latency_improvement=args.min_latency_improvement,
        candidate_prefix=args.candidate_prefix,
    )
    print(json.dumps(_decision_summary(decision), indent=2, sort_keys=True))
    if not decision.promoted:
        sys.exit(1)


def _decision_summary(decision: PromotionDecision) -> dict[str, object]:
    return {
        "candidate_model_version": decision.candidate_model_version,
        "teacher_model_version": decision.teacher_model_version,
        "promoted": decision.promoted,
        "ndcg_drop": decision.ndcg_drop,
        "latency_improvement": decision.latency_improvement,
        "failed_stages": [
            {"name": stage.name, "detail": stage.detail}
            for stage in decision.stages
            if not stage.passed
        ],
    }


if __name__ == "__main__":
    main()
