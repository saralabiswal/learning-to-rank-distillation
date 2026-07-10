"""Response-based knowledge distillation for ranking students."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from learning_to_rank_distillation.distillation.common import (
    TrainingHistory,
    listwise_label_loss,
    query_batches,
)
from learning_to_rank_distillation.models.student import StudentConfig, TwoTowerStudent
from learning_to_rank_distillation.schema import RankingExample


def response_based_distillation_loss(
    *,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """KL-divergence response matching blended with label ranking loss.

    Implements FR-2.2.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    kd_loss = F.kl_div(
        F.log_softmax(student_scores / temperature, dim=0),
        F.softmax(teacher_scores / temperature, dim=0),
        reduction="batchmean",
    ) * (temperature**2)
    supervised_loss = listwise_label_loss(student_scores, labels)
    return alpha * kd_loss + (1.0 - alpha) * supervised_loss


def train_response_based_student(
    examples: list[RankingExample],
    teacher_scores: list[float] | np.ndarray,
    *,
    config: StudentConfig | None = None,
    epochs: int = 8,
    learning_rate: float = 0.01,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> tuple[TwoTowerStudent, TrainingHistory]:
    """Train a response-distilled two-tower student.

    Implements FR-2.2 and FR-2.4.
    """

    if len(examples) != len(teacher_scores):
        raise ValueError("teacher_scores must align one-to-one with examples")
    score_by_key = {
        (example.query_id, example.item_id): float(score)
        for example, score in zip(examples, teacher_scores, strict=True)
    }

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
            teacher_tensor = torch.from_numpy(
                np.asarray(
                    [score_by_key[(example.query_id, example.item_id)] for example in batch],
                    dtype=np.float32,
                )
            )
            query_features, item_features = student.tensors(batch)
            optimizer.zero_grad()
            scores = student.model(query_features, item_features)
            loss = response_based_distillation_loss(
                student_scores=scores,
                teacher_scores=teacher_tensor,
                labels=labels,
                temperature=temperature,
                alpha=alpha,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return student, TrainingHistory(losses)
