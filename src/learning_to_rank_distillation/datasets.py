"""Dataset selection helpers for CLI and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from learning_to_rank_distillation.adapters.esci import ESCIAdapter
from learning_to_rank_distillation.adapters.rectour import RecTourAdapter
from learning_to_rank_distillation.adapters.synthetic import make_synthetic_ranking_data
from learning_to_rank_distillation.schema import RankingExample

DatasetName = Literal["synthetic", "esci", "rectour"]


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset-loading options shared by commands."""

    name: DatasetName = "synthetic"
    data_dir: Path | None = None
    num_queries: int = 36
    items_per_query: int = 8
    seed: int = 13
    limit: int | None = None
    split: str | None = "train"


def load_ranking_examples(config: DatasetConfig) -> list[RankingExample]:
    """Load examples from the selected dataset adapter."""

    if config.name == "synthetic":
        return make_synthetic_ranking_data(
            num_queries=config.num_queries,
            items_per_query=config.items_per_query,
            seed=config.seed,
        )
    if config.name == "esci":
        return ESCIAdapter(
            data_dir=config.data_dir or Path("data/esci"),
            split=config.split,
        ).load(limit=config.limit)
    if config.name == "rectour":
        return RecTourAdapter(config.data_dir or Path("data/rectour")).load(limit=config.limit)
    raise ValueError(f"unsupported dataset: {config.name}")
