from pathlib import Path

from learning_to_rank_distillation.distillation.no_kd_baseline import train_no_kd_student
from learning_to_rank_distillation.models.student import StudentConfig
from learning_to_rank_distillation.production.index_lifecycle import (
    build_student_bundle_version,
    publish_bundle_version,
    resolve_published_bundle,
    validate_student_bundle,
)
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_index_lifecycle_builds_validates_and_publishes_bundle(tmp_path: Path) -> None:
    examples = make_synthetic_ranking_data(num_queries=5, items_per_query=4, seed=43)
    student, _ = train_no_kd_student(
        examples,
        config=StudentConfig(embedding_dim=8, random_state=43),
        epochs=1,
    )

    result = build_student_bundle_version(student, examples, tmp_path / "registry")
    validate_student_bundle(result.version_dir)
    current_path = publish_bundle_version(tmp_path / "registry", result.version_dir)
    resolved = resolve_published_bundle(tmp_path / "registry")

    assert result.item_count > 0
    assert current_path.exists()
    assert resolved == result.version_dir.resolve()
