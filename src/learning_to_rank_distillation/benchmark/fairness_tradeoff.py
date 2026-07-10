"""Fairness tradeoff benchmark."""

from __future__ import annotations

import json
import os
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [row.low_exposure_impression_share for row in rows]
    y = [row.ndcg_at_5 for row in rows]
    labels = [f"{row.exposure_floor:.1f}" for row in rows]
    ax.plot(x, y, marker="o")
    for label, x_value, y_value in zip(labels, x, y, strict=True):
        ax.annotate(label, (x_value, y_value), fontsize=8)
    ax.set_xlabel("Top-5 low-exposure impression share")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Relevance vs. exposure fairness")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _configure_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = (output_dir / ".matplotlib-cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
