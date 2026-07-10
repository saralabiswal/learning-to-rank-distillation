from pathlib import Path

from fastapi.testclient import TestClient

from learning_to_rank_distillation.distillation.no_kd_baseline import train_no_kd_student
from learning_to_rank_distillation.models.student import StudentConfig
from learning_to_rank_distillation.production.bundle import (
    load_student_bundle,
    save_student_bundle,
)
from learning_to_rank_distillation.production.serving import create_app
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_serving_app_ranks_from_loaded_bundle(tmp_path: Path) -> None:
    examples = make_synthetic_ranking_data(num_queries=5, items_per_query=4, seed=47)
    student, _ = train_no_kd_student(
        examples,
        config=StudentConfig(embedding_dim=8, random_state=47),
        epochs=1,
    )
    bundle_path = save_student_bundle(student, examples, tmp_path / "bundle")
    bundle = load_student_bundle(bundle_path)
    client = TestClient(create_app(bundle=bundle))

    health = client.get("/health")
    response = client.post(
        "/rank",
        json={"query_features": examples[0].features, "k": 3},
    )
    metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["item_count"] > 0
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3
    assert metrics.status_code == 200
    assert "ltrd_rank_requests_total" in metrics.text


def test_serving_app_reports_unconfigured_bundle() -> None:
    client = TestClient(create_app())

    health = client.get("/health")
    response = client.post("/rank", json={"query_features": {}, "k": 1})

    assert health.json()["status"] == "unconfigured"
    assert response.status_code == 503
