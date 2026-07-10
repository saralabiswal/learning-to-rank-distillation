"""Scalarized Pareto search benchmark for exposure fairness."""

from __future__ import annotations

import json
import os
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [row.low_exposure_impression_share for row in rows]
    y = [row.ndcg_at_5 for row in rows]
    colors = ["tab:green" if row.is_pareto_efficient else "tab:gray" for row in rows]
    ax.scatter(x, y, c=colors, s=52)
    for row, x_value, y_value in zip(rows, x, y, strict=True):
        ax.annotate(f"{row.fairness_weight:g}", (x_value, y_value), fontsize=8)
    ax.set_xlabel("Top-5 low-exposure impression share")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Scalarized relevance/fairness Pareto search")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _configure_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = (output_dir / ".matplotlib-cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
