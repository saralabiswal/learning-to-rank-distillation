"""Scalarized Pareto search benchmark for exposure fairness."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from learning_to_rank_distillation.fairness.pareto import (
    ParetoSearchRow,
    pareto_frontier_search,
)
from learning_to_rank_distillation.schema import RankingExample


def run_fairness_pareto_search(
    *,
    train_examples: list[RankingExample],
    eval_examples: list[RankingExample],
    relevance_scores: list[float],
    fairness_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    output_dir: Path = Path("artifacts"),
) -> list[ParetoSearchRow]:
    """Run a scalarization sweep and write Pareto-search artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_cache(output_dir)
    rows = pareto_frontier_search(
        train_examples=train_examples,
        eval_examples=eval_examples,
        relevance_scores=relevance_scores,
        fairness_weights=fairness_weights,
    )
    _write_outputs(rows, output_dir)
    return rows


def _write_outputs(rows: list[ParetoSearchRow], output_dir: Path) -> None:
    (output_dir / "fairness_pareto_frontier.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot(rows, output_dir / "fairness_pareto_frontier.png")


def _plot(rows: list[ParetoSearchRow], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = [row.low_exposure_impression_share for row in rows]
    y = [row.ndcg_at_5 for row in rows]
    ax.plot(x, y, color="#b7c0cb", linewidth=1.2, zorder=1)
    grouped = _group_by_coordinate(rows)
    for index, ((x_value, y_value), grouped_rows) in enumerate(grouped.items()):
        is_efficient = any(row.is_pareto_efficient for row in grouped_rows)
        color = "#1f7a4d" if is_efficient else "#777f8a"
        ax.scatter(
            x_value,
            y_value,
            c=color,
            s=76 + 16 * (len(grouped_rows) - 1),
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        ax.annotate(
            _weight_label([row.fairness_weight for row in grouped_rows], is_efficient),
            (x_value, y_value),
            xytext=_pareto_label_offset([row.fairness_weight for row in grouped_rows], index),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="semibold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            arrowprops={"arrowstyle": "-", "color": "#8a94a3", "lw": 0.7},
            zorder=4,
        )
    _pad_axes(ax, x, y)
    ax.set_xlabel("Top-5 low-exposure impression share")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Scalarized relevance/fairness Pareto search")
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        0.02,
        "Green points are Pareto-efficient; gray points are dominated on this benchmark run.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#5c6875",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _group_by_coordinate(
    rows: list[ParetoSearchRow],
) -> dict[tuple[float, float], list[ParetoSearchRow]]:
    grouped: dict[tuple[float, float], list[ParetoSearchRow]] = defaultdict(list)
    for row in rows:
        key = (round(row.low_exposure_impression_share, 6), round(row.ndcg_at_5, 6))
        grouped[key].append(row)
    return dict(grouped)


def _weight_label(values: list[float], is_efficient: bool) -> str:
    labels = [f"{value:g}" for value in values]
    prefix = "Pareto " if is_efficient else ""
    if len(labels) == 1:
        return f"{prefix}w={labels[0]}"
    return f"{prefix}w={','.join(labels)}"


def _pareto_label_offset(values: list[float], index: int) -> tuple[int, int]:
    value_set = {round(value, 6) for value in values}
    if 4.0 in value_set:
        return (12, -4)
    if 2.0 in value_set:
        return (-84, -24)
    if 1.0 in value_set:
        return (-102, 16)
    if 0.5 in value_set:
        return (-112, -8)
    if {0.0, 0.25}.issubset(value_set):
        return (12, 14)
    offsets = [(12, 12), (12, -22), (-84, 12), (-84, -22)]
    return offsets[index % len(offsets)]


def _pad_axes(ax: object, x_values: list[float], y_values: list[float]) -> None:
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.18, 0.025)
    y_pad = max((y_max - y_min) * 0.25, 0.02)
    ax.set_xlim(max(0.0, x_min - x_pad), min(1.0, x_max + x_pad))
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))


def _configure_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = (output_dir / ".matplotlib-cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
