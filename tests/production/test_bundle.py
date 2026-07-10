from pathlib import Path

from learning_to_rank_distillation.distillation.no_kd_baseline import train_no_kd_student
from learning_to_rank_distillation.models.student import StudentConfig
from learning_to_rank_distillation.production.bundle import (
    load_student_bundle,
    save_student_bundle,
)
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_student_bundle_round_trips_and_builds_item_index(tmp_path: Path) -> None:
    examples = make_synthetic_ranking_data(num_queries=5, items_per_query=4, seed=41)
    student, _ = train_no_kd_student(
        examples,
        config=StudentConfig(embedding_dim=8, random_state=41),
        epochs=1,
    )

    bundle_path = save_student_bundle(
        student,
        examples,
        tmp_path / "bundle",
        metrics={"ndcg_at_5": 0.5},
        training_config={"epochs": 1},
    )
    bundle = load_student_bundle(bundle_path)
    item_index = bundle.item_index()
    query_embeddings = bundle.student.query_embeddings(examples[:2])
    results = item_index.search(query_embeddings, k=3)

    assert (bundle_path / "metadata.json").exists()
    assert bundle.metadata["metrics"] == {"ndcg_at_5": 0.5}
    assert bundle.item_embeddings.shape[1] == 8
    assert len(results) == 2
    assert all(len(row) == 3 for row in results)
