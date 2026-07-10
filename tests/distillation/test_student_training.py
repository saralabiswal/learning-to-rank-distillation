import numpy as np
import torch

from learning_to_rank_distillation.distillation.feature_based import (
    feature_based_distillation_loss,
    project_teacher_representations,
    train_feature_based_student,
)
from learning_to_rank_distillation.distillation.no_kd_baseline import train_no_kd_student
from learning_to_rank_distillation.distillation.relation_based import (
    relation_based_distillation_loss,
    train_relation_based_student,
)
from learning_to_rank_distillation.distillation.response_based import (
    response_based_distillation_loss,
    train_response_based_student,
)
from learning_to_rank_distillation.models.ann import FaissItemIndex
from learning_to_rank_distillation.models.student import StudentConfig, TwoTowerStudent
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_two_tower_student_predicts_and_builds_ann_index() -> None:
    examples = make_synthetic_ranking_data(num_queries=5, items_per_query=4, seed=4)
    student = TwoTowerStudent(StudentConfig(embedding_dim=8, random_state=4)).initialize(examples)

    scores = student.predict(examples)
    index = FaissItemIndex.from_student(student, examples)
    query_embeddings = student.query_embeddings(examples[:2])
    results = index.search(query_embeddings, k=3)

    assert scores.shape == (len(examples),)
    assert len(results) == 2
    assert all(len(row) == 3 for row in results)


def test_no_kd_baseline_trains_student() -> None:
    examples = make_synthetic_ranking_data(num_queries=6, items_per_query=4, seed=5)

    student, history = train_no_kd_student(
        examples,
        config=StudentConfig(embedding_dim=8, random_state=5),
        epochs=2,
    )

    assert len(history.losses) == 2
    assert np.isfinite(history.final_loss)
    assert student.predict(examples).shape == (len(examples),)


def test_response_based_distillation_trains_student() -> None:
    examples = make_synthetic_ranking_data(num_queries=6, items_per_query=4, seed=6)
    teacher_scores = np.linspace(0.0, 1.0, len(examples), dtype=np.float32)

    student, history = train_response_based_student(
        examples,
        teacher_scores,
        config=StudentConfig(embedding_dim=8, random_state=6),
        epochs=2,
    )

    assert len(history.losses) == 2
    assert np.isfinite(history.final_loss)
    assert student.predict(examples).shape == (len(examples),)


def test_response_based_distillation_loss_is_finite() -> None:
    loss = response_based_distillation_loss(
        student_scores=torch.tensor([0.1, 0.2, 0.3]),
        teacher_scores=torch.tensor([0.3, 0.2, 0.1]),
        labels=torch.tensor([0.0, 1.0, 2.0]),
        temperature=2.0,
    )

    assert torch.isfinite(loss)


def test_feature_based_distillation_trains_student() -> None:
    examples = make_synthetic_ranking_data(num_queries=6, items_per_query=4, seed=7)
    teacher_representations = np.column_stack(
        [
            np.linspace(0.0, 1.0, len(examples), dtype=np.float32),
            np.asarray([example.label for example in examples], dtype=np.float32),
        ]
    )

    student, history = train_feature_based_student(
        examples,
        teacher_representations,
        config=StudentConfig(embedding_dim=8, random_state=7),
        epochs=2,
    )

    assert len(history.losses) == 2
    assert np.isfinite(history.final_loss)
    assert student.predict(examples).shape == (len(examples),)


def test_feature_based_distillation_loss_is_finite() -> None:
    loss = feature_based_distillation_loss(
        student_scores=torch.tensor([0.1, 0.2, 0.3]),
        student_item_embeddings=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        teacher_item_embeddings=torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]),
        labels=torch.tensor([0.0, 1.0, 2.0]),
    )

    assert torch.isfinite(loss)


def test_project_teacher_representations_matches_embedding_dim() -> None:
    teacher_representations = np.arange(15, dtype=np.float32).reshape(5, 3)

    projected = project_teacher_representations(teacher_representations, embedding_dim=8)

    assert projected.shape == (5, 8)
    assert np.isfinite(projected).all()


def test_relation_based_distillation_trains_student() -> None:
    examples = make_synthetic_ranking_data(num_queries=6, items_per_query=4, seed=8)
    teacher_scores = np.linspace(1.0, 0.0, len(examples), dtype=np.float32)

    student, history = train_relation_based_student(
        examples,
        teacher_scores,
        config=StudentConfig(embedding_dim=8, random_state=8),
        epochs=2,
    )

    assert len(history.losses) == 2
    assert np.isfinite(history.final_loss)
    assert student.predict(examples).shape == (len(examples),)


def test_relation_based_distillation_loss_is_finite() -> None:
    loss = relation_based_distillation_loss(
        student_scores=torch.tensor([0.1, 0.2, 0.3]),
        teacher_scores=torch.tensor([0.3, 0.2, 0.1]),
        labels=torch.tensor([0.0, 1.0, 2.0]),
    )

    assert torch.isfinite(loss)
