"""Fairness tradeoff benchmark."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from learning_to_rank_distillation.benchmark.metrics import ndcg_at_k
from learning_to_rank_distillation.fairness.constrained_rerank import (
    ranking_position_scores,
    rerank_by_query,
)
from learning_to_rank_distillation.fairness.exposure import (
    compute_exposure_stats,
    exposure_gini,
    low_exposure_groups,
    low_exposure_impression_share,
)
from learning_to_rank_distillation.schema import RankingExample


@dataclass(frozen=True, slots=True)
class FairnessTradeoffRow:
    exposure_floor: float
    ndcg_at_5: float
    low_exposure_impression_share: float
    exposure_gini_at_5: float


def run_fairness_tradeoff(
    *,
    train_examples: list[RankingExample],
    eval_examples: list[RankingExample],
    relevance_scores: list[float],
    exposure_floors: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4),
    output_dir: Path = Path("artifacts"),
) -> list[FairnessTradeoffRow]:
    """Sweep exposure floors and report relevance/fairness tradeoffs.

    Implements FR-4.3.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_cache(output_dir)
    stats = compute_exposure_stats(train_examples)
    low_groups = low_exposure_groups(stats)
    rows: list[FairnessTradeoffRow] = []
    for floor in exposure_floors:
        reranked = rerank_by_query(
            eval_examples,
            relevance_scores,
            stats,
            exposure_floor=floor,
        )
        position_scores = ranking_position_scores(reranked)
        rows.append(
            FairnessTradeoffRow(
                exposure_floor=floor,
                ndcg_at_5=ndcg_at_k(reranked, position_scores, k=5),
                low_exposure_impression_share=low_exposure_impression_share(
                    reranked,
                    low_groups,
                    top_k=5,
                ),
                exposure_gini_at_5=exposure_gini(reranked, top_k=5),
            )
        )
    _write_outputs(rows, output_dir)
    return rows


def _write_outputs(rows: list[FairnessTradeoffRow], output_dir: Path) -> None:
    (output_dir / "fairness_tradeoff.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot(rows, output_dir / "fairness_tradeoff.png")


def _plot(rows: list[FairnessTradeoffRow], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = [row.low_exposure_impression_share for row in rows]
    y = [row.ndcg_at_5 for row in rows]
    ax.plot(x, y, color="#b7c0cb", linewidth=1.2, zorder=1)
    grouped = _group_by_coordinate(rows)
    for index, ((x_value, y_value), grouped_rows) in enumerate(grouped.items()):
        ax.scatter(
            x_value,
            y_value,
            s=76 + 16 * (len(grouped_rows) - 1),
            c="#0b6f6a",
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        ax.annotate(
            _floor_label([row.exposure_floor for row in grouped_rows]),
            (x_value, y_value),
            xytext=(12, 14 if index % 2 == 0 else -22),
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
    ax.set_title("Relevance vs. exposure fairness")
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    if len(grouped) == 1:
        ax.text(
            0.01,
            0.02,
            "All tested exposure floors produced the same top-5 ranking on this smoke fixture.",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#5c6875",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _group_by_coordinate(
    rows: list[FairnessTradeoffRow],
) -> dict[tuple[float, float], list[FairnessTradeoffRow]]:
    grouped: dict[tuple[float, float], list[FairnessTradeoffRow]] = defaultdict(list)
    for row in rows:
        key = (round(row.low_exposure_impression_share, 6), round(row.ndcg_at_5, 6))
        grouped[key].append(row)
    return dict(grouped)


def _floor_label(values: list[float]) -> str:
    labels = [f"{value:.1f}" for value in values]
    if len(labels) == 1:
        return f"floor {labels[0]}"
    return f"floors {labels[0]}-{labels[-1]}"


def _pad_axes(ax: object, x_values: list[float], y_values: list[float]) -> None:
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.18, 0.035)
    y_pad = max((y_max - y_min) * 0.25, 0.02)
    ax.set_xlim(max(0.0, x_min - x_pad), min(1.0, x_max + x_pad))
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))


def _configure_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = (output_dir / ".matplotlib-cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
