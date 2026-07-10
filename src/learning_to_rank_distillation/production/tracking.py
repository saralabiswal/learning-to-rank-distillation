"""Experiment tracking helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def log_experiment_run(
    *,
    run_name: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    tracking_path: Path = Path("artifacts/experiments.jsonl"),
) -> Path:
    """Log an experiment run to MLflow when configured, otherwise JSONL."""

    if os.getenv("LTRD_TRACKING_BACKEND") == "mlflow" and _try_log_mlflow(
        run_name=run_name,
        metrics=metrics,
        params=params,
    ):
        return tracking_path

    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_name": run_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "params": params,
    }
    with tracking_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")
    return tracking_path


def benchmark_metrics_from_rows(rows: list[Any]) -> dict[str, float]:
    """Flatten benchmark rows into tracker-friendly metric names."""

    metrics: dict[str, float] = {}
    for row in rows:
        values = asdict(row)
        model = str(values.pop("model")).replace("-", "_")
        for name, value in values.items():
            if isinstance(value, int | float):
                metrics[f"{model}.{name}"] = float(value)
    return metrics


def _try_log_mlflow(
    *,
    run_name: str,
    metrics: dict[str, float],
    params: dict[str, Any],
) -> bool:
    try:
        import mlflow
    except ModuleNotFoundError:
        return False

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    return True
