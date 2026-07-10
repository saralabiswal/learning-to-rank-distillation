"""Relation-based knowledge distillation for ranking students."""

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


def relation_based_distillation_loss(
    *,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = 0.6,
) -> torch.Tensor:
    """Blend label ranking loss with pairwise teacher/student relation matching."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape")

    student_relations = _pairwise_relations(student_scores, temperature=temperature)
    teacher_relations = _pairwise_relations(teacher_scores, temperature=temperature)
    relation_loss = F.mse_loss(student_relations, teacher_relations)
    supervised_loss = listwise_label_loss(student_scores, labels)
    return alpha * relation_loss + (1.0 - alpha) * supervised_loss


def train_relation_based_student(
    examples: list[RankingExample],
    teacher_scores: list[float] | np.ndarray,
    *,
    config: StudentConfig | None = None,
    epochs: int = 8,
    learning_rate: float = 0.01,
    temperature: float = 1.0,
    alpha: float = 0.6,
) -> tuple[TwoTowerStudent, TrainingHistory]:
    """Train a student to preserve teacher pairwise/listwise ranking relations."""

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
            loss = relation_based_distillation_loss(
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


def _pairwise_relations(scores: torch.Tensor, *, temperature: float) -> torch.Tensor:
    differences = scores[:, None] - scores[None, :]
    return torch.tanh(differences / temperature)
