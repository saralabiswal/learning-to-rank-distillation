"""End-to-end benchmark runner.

Author: Sarala Biswal
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from learning_to_rank_distillation.benchmark.fairness_pareto import run_fairness_pareto_search
from learning_to_rank_distillation.benchmark.fairness_tradeoff import run_fairness_tradeoff
from learning_to_rank_distillation.benchmark.metrics import ndcg_at_k
from learning_to_rank_distillation.data import split_by_query
from learning_to_rank_distillation.datasets import DatasetConfig, DatasetName, load_ranking_examples
from learning_to_rank_distillation.distillation.no_kd_baseline import train_no_kd_student
from learning_to_rank_distillation.distillation.response_based import train_response_based_student
from learning_to_rank_distillation.governance.promotion_gate import ModelMetrics, PromotionGate
from learning_to_rank_distillation.models.student import StudentConfig, TwoTowerStudent
from learning_to_rank_distillation.models.teacher import LightGBMLambdaMARTTeacher, content_hash
from learning_to_rank_distillation.production.tracking import (
    benchmark_metrics_from_rows,
    log_experiment_run,
)
from learning_to_rank_distillation.schema import RankingExample


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    model: str
    ndcg_at_5: float
    ndcg_at_10: float
    model_size_bytes: int
    latency_p50_ms: float
    latency_p99_ms: float


def run_benchmark(
    *,
    dataset: DatasetName = "synthetic",
    data_dir: Path | None = None,
    num_queries: int = 36,
    items_per_query: int = 8,
    student_embedding_dims: Sequence[int] = (8, 16, 32),
    student_epochs: int = 4,
    output_dir: Path = Path("artifacts"),
    seed: int = 13,
    limit: int | None = None,
    split: str | None = "train",
) -> list[BenchmarkRow]:
    """Run the v1.0 benchmark on the selected ranking dataset.

    Implements FR-2.4, FR-3.1, and FR-3.2.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_cache(output_dir)
    examples = load_ranking_examples(
        DatasetConfig(
            name=dataset,
            data_dir=data_dir,
            num_queries=num_queries,
            items_per_query=items_per_query,
            seed=seed,
            limit=limit,
            split=split,
        )
    )
    query_split = split_by_query(examples, seed=seed)

    teacher = LightGBMLambdaMARTTeacher(random_state=seed, n_estimators=30).fit(query_split.train)
    teacher_metadata = teacher.save(output_dir / "models", query_split.train)
    teacher_scores = teacher.predict(query_split.test)
    teacher_train_scores = teacher.predict(query_split.train)
    rows = [
        BenchmarkRow(
            model="teacher-lightgbm",
            ndcg_at_5=ndcg_at_k(query_split.test, teacher_scores, k=5),
            ndcg_at_10=ndcg_at_k(query_split.test, teacher_scores, k=10),
            model_size_bytes=_teacher_size_bytes(teacher_metadata),
            **_latency(lambda: teacher.predict(query_split.test)),
        )
    ]

    no_kd_student, _ = train_no_kd_student(
        query_split.train,
        config=StudentConfig(embedding_dim=16, random_state=seed),
        epochs=student_epochs,
    )
    rows.append(_student_row("student-no-kd-d16", no_kd_student, query_split.test))

    for embedding_dim in student_embedding_dims:
        kd_student, _ = train_response_based_student(
            query_split.train,
            teacher_train_scores,
            config=StudentConfig(embedding_dim=embedding_dim, random_state=seed),
            epochs=student_epochs,
        )
        rows.append(_student_row(f"student-kd-d{embedding_dim}", kd_student, query_split.test))

    run_fairness_tradeoff(
        train_examples=query_split.train,
        eval_examples=query_split.test,
        relevance_scores=teacher_scores.tolist(),
        output_dir=output_dir,
    )
    run_fairness_pareto_search(
        train_examples=query_split.train,
        eval_examples=query_split.test,
        relevance_scores=teacher_scores.tolist(),
        output_dir=output_dir,
    )
    _log_promotion_decision(rows, query_split.train, output_dir)
    _write_outputs(rows, output_dir)
    log_experiment_run(
        run_name=f"benchmark-{dataset}",
        metrics=benchmark_metrics_from_rows(rows),
        params={
            "dataset": dataset,
            "num_queries": num_queries,
            "items_per_query": items_per_query,
            "student_epochs": student_epochs,
            "seed": seed,
            "limit": limit,
            "split": split,
        },
        tracking_path=output_dir / "experiments.jsonl",
    )
    return rows


def _student_row(
    model_name: str,
    student: TwoTowerStudent,
    examples: list[RankingExample],
) -> BenchmarkRow:
    scores = student.predict(examples)
    return BenchmarkRow(
        model=model_name,
        ndcg_at_5=ndcg_at_k(examples, scores, k=5),
        ndcg_at_10=ndcg_at_k(examples, scores, k=10),
        model_size_bytes=student.estimated_size_bytes(),
        **_latency(lambda: student.predict(examples)),
    )


def _latency(callback: Callable[[], object], *, repeats: int = 20) -> dict[str, float]:
    durations_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "latency_p50_ms": float(np.percentile(durations_ms, 50)),
        "latency_p99_ms": float(np.percentile(durations_ms, 99)),
    }


def _teacher_size_bytes(metadata_path: Path) -> int:
    prefix = metadata_path.stem
    return sum(path.stat().st_size for path in metadata_path.parent.glob(f"{prefix}.*"))


def _write_outputs(rows: list[BenchmarkRow], output_dir: Path) -> None:
    table_path = output_dir / "benchmark_table.json"
    table_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot_quality_latency(rows, output_dir / "quality_latency_pareto.png")


def _log_promotion_decision(
    rows: list[BenchmarkRow],
    train_examples: list[RankingExample],
    output_dir: Path,
) -> None:
    teacher = next(row for row in rows if row.model == "teacher-lightgbm")
    kd_candidates = [row for row in rows if row.model.startswith("student-kd")]
    if not kd_candidates:
        return
    candidate = max(kd_candidates, key=lambda row: row.ndcg_at_5)
    data_hash = content_hash(train_examples)
    PromotionGate(registry_path=output_dir / "promotion_registry.sqlite").evaluate(
        candidate=ModelMetrics(
            model_version=candidate.model,
            ndcg_at_5=candidate.ndcg_at_5,
            ndcg_at_10=candidate.ndcg_at_10,
            latency_p99_ms=candidate.latency_p99_ms,
            data_hash=data_hash,
        ),
        teacher=ModelMetrics(
            model_version=teacher.model,
            ndcg_at_5=teacher.ndcg_at_5,
            ndcg_at_10=teacher.ndcg_at_10,
            latency_p99_ms=teacher.latency_p99_ms,
            data_hash=data_hash,
        ),
    )


def _plot_quality_latency(rows: list[BenchmarkRow], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for row in rows:
        color, marker = _model_style(row.model)
        ax.scatter(
            row.latency_p99_ms,
            row.ndcg_at_5,
            s=72,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            _short_model_name(row.model),
            (row.latency_p99_ms, row.ndcg_at_5),
            xytext=_model_label_offset(row.model),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="semibold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#8a94a3",
                "lw": 0.7,
                "shrinkA": 0,
                "shrinkB": 4,
            },
            zorder=4,
        )
    _pad_axes(
        ax,
        [row.latency_p99_ms for row in rows],
        [row.ndcg_at_5 for row in rows],
    )
    ax.set_xlabel("p99 latency (ms)")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Quality vs. latency trade-off")
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        0.02,
        "Upper-left is better: higher quality at lower p99 latency.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#5c6875",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _model_style(model: str) -> tuple[str, str]:
    if model == "teacher-lightgbm":
        return "#315a9a", "D"
    if "no-kd" in model:
        return "#9a5b00", "o"
    return "#0b6f6a", "o"


def _short_model_name(model: str) -> str:
    return (
        model.replace("teacher-lightgbm", "Teacher")
        .replace("student-no-kd-d16", "No KD d16")
        .replace("student-kd-d", "KD d")
    )


def _model_label_offset(model: str) -> tuple[int, int]:
    offsets = {
        "teacher-lightgbm": (-112, -2),
        "student-no-kd-d16": (8, 10),
        "student-kd-d8": (8, -18),
        "student-kd-d16": (-74, -18),
        "student-kd-d32": (-74, 10),
    }
    return offsets.get(model, (8, 8))


def _pad_axes(ax: object, x_values: list[float], y_values: list[float]) -> None:
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.18, 0.01)
    y_pad = max((y_max - y_min) * 0.25, 0.01)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _configure_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = (output_dir / ".matplotlib-cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


def format_markdown_table(rows: list[BenchmarkRow]) -> str:
    header = (
        "| model | NDCG@5 | NDCG@10 | size bytes | p50 ms | p99 ms |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| "
            f"{row.model} | {row.ndcg_at_5:.4f} | {row.ndcg_at_10:.4f} | "
            f"{row.model_size_bytes} | {row.latency_p50_ms:.3f} | "
            f"{row.latency_p99_ms:.3f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    # This parser mirrors the Makefile and CI smoke benchmark so local and
    # automated runs exercise the same dataset/configuration surface.
    parser = argparse.ArgumentParser(description="Run the v1.0 LTRD benchmark.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "esci", "rectour", "movielens"],
        default="synthetic",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--num-queries", type=int, default=36)
    parser.add_argument("--items-per-query", type=int, default=8)
    parser.add_argument("--student-epochs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)

    # Delegate training/evaluation to run_benchmark; main stays a thin wrapper
    # that prints a human-readable table for terminal users.
    rows = run_benchmark(
        dataset=args.dataset,
        data_dir=args.data_dir,
        num_queries=args.num_queries,
        items_per_query=args.items_per_query,
        student_epochs=args.student_epochs,
        output_dir=args.output_dir,
        limit=args.limit,
        split=args.split,
    )
    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()
