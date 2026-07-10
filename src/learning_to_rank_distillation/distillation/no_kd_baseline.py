"""No-distillation student baseline."""

from __future__ import annotations

import numpy as np
import torch

from learning_to_rank_distillation.distillation.common import (
    TrainingHistory,
    listwise_label_loss,
    query_batches,
)
from learning_to_rank_distillation.models.student import StudentConfig, TwoTowerStudent
from learning_to_rank_distillation.schema import RankingExample


def train_no_kd_student(
    examples: list[RankingExample],
    *,
    config: StudentConfig | None = None,
    epochs: int = 8,
    learning_rate: float = 0.01,
) -> tuple[TwoTowerStudent, TrainingHistory]:
    """Train the two-tower student on labels only.

    Implements FR-2.3.
    """

    student = TwoTowerStudent(config or StudentConfig()).initialize(examples)
    if student.model is None:
        raise RuntimeError("student failed to initialize")

    optimizer = torch.optim.Adam(student.model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        epoch_losses: list[float] = []
        student.model.train()
        for batch in query_batches(examples):
            labels = torch.from_numpy(
                np.asarray([example.label for example in batch], dtype=np.float32)
            )
            query_features, item_features = student.tensors(batch)
            optimizer.zero_grad()
            scores = student.model(query_features, item_features)
            loss = listwise_label_loss(scores, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return student, TrainingHistory(losses)
