"""Feature-based knowledge distillation for ranking students."""

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


def feature_based_distillation_loss(
    *,
    student_scores: torch.Tensor,
    student_item_embeddings: torch.Tensor,
    teacher_item_embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Blend label ranking loss with item-representation matching.

    This is intentionally representation-agnostic: transformer teachers can pass hidden states
    later, while current tests use deterministic teacher-side representations.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if student_item_embeddings.shape != teacher_item_embeddings.shape:
        raise ValueError("student and teacher embeddings must have the same shape")

    representation_loss = F.mse_loss(
        F.normalize(student_item_embeddings, p=2, dim=1),
        F.normalize(teacher_item_embeddings, p=2, dim=1),
    )
    supervised_loss = listwise_label_loss(student_scores, labels)
    return alpha * representation_loss + (1.0 - alpha) * supervised_loss


def train_feature_based_student(
    examples: list[RankingExample],
    teacher_item_representations: np.ndarray,
    *,
    config: StudentConfig | None = None,
    epochs: int = 8,
    learning_rate: float = 0.01,
    alpha: float = 0.5,
) -> tuple[TwoTowerStudent, TrainingHistory]:
    """Train a student by matching item tower embeddings to teacher representations."""

    student_config = config or StudentConfig()
    teacher_targets = project_teacher_representations(
        teacher_item_representations,
        embedding_dim=student_config.embedding_dim,
    )
    if len(examples) != len(teacher_targets):
        raise ValueError("teacher_item_representations must align one-to-one with examples")
    target_by_key = {
        (example.query_id, example.item_id): target
        for example, target in zip(examples, teacher_targets, strict=True)
    }

    student = TwoTowerStudent(student_config).initialize(examples)
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
                    [target_by_key[(example.query_id, example.item_id)] for example in batch],
                    dtype=np.float32,
                )
            )
            query_features, item_features = student.tensors(batch)
            optimizer.zero_grad()
            scores = student.model(query_features, item_features)
            item_embeddings = student.model.encode_item(item_features)
            loss = feature_based_distillation_loss(
                student_scores=scores,
                student_item_embeddings=item_embeddings,
                teacher_item_embeddings=teacher_tensor,
                labels=labels,
                alpha=alpha,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return student, TrainingHistory(losses)


def project_teacher_representations(
    teacher_representations: np.ndarray,
    *,
    embedding_dim: int,
) -> np.ndarray:
    """Project teacher representations to the student embedding dimension.

    The projection is deterministic so benchmark runs remain reproducible. If the teacher already
    emits the student dimension, values are only normalized for scale.
    """

    if embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")
    representations = np.asarray(teacher_representations, dtype=np.float32)
    if representations.ndim == 1:
        representations = representations[:, None]
    if representations.ndim != 2:
        raise ValueError("teacher_representations must be a 1D or 2D array")
    if representations.shape[0] == 0:
        raise ValueError("teacher_representations must be non-empty")

    centered = representations - representations.mean(axis=0, keepdims=True)
    if centered.shape[1] > embedding_dim:
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        projected = np.zeros((centered.shape[0], embedding_dim), dtype=np.float32)
        component_count = min(embedding_dim, components.shape[0])
        projected[:, :component_count] = centered @ components[:component_count].T
    else:
        projected = np.zeros((centered.shape[0], embedding_dim), dtype=np.float32)
        projected[:, : centered.shape[1]] = centered

    scale = projected.std(axis=0, keepdims=True)
    projected = np.divide(projected, scale, out=np.zeros_like(projected), where=scale > 0)
    return projected.astype(np.float32)
