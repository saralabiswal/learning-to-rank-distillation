"""Shared training utilities for student models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from learning_to_rank_distillation.schema import RankingExample, group_by_query


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    losses: list[float]

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def query_batches(examples: list[RankingExample]) -> Iterable[list[RankingExample]]:
    """Yield one ranked list per query."""

    yield from group_by_query(examples).values()


def label_distribution(labels: torch.Tensor) -> torch.Tensor:
    """Convert graded relevance labels into a listwise probability target."""

    return F.softmax(labels.float(), dim=0)


def listwise_label_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Listwise ranking loss against ground-truth labels."""

    target = label_distribution(labels)
    return -(target * F.log_softmax(scores, dim=0)).sum()
