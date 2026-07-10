from pathlib import Path

from learning_to_rank_distillation.data import split_by_query
from learning_to_rank_distillation.models.teacher import LightGBMLambdaMARTTeacher, content_hash
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_lightgbm_teacher_trains_predicts_and_saves(tmp_path: Path) -> None:
    examples = make_synthetic_ranking_data(num_queries=10, items_per_query=5, seed=3)
    split = split_by_query(examples, seed=3)
    teacher = LightGBMLambdaMARTTeacher(n_estimators=5, random_state=3)

    teacher.fit(split.train)
    scores = teacher.predict(split.test)
    metrics = teacher.evaluate(split.test)
    metadata_path = teacher.save(tmp_path, split.train)

    assert len(scores) == len(split.test)
    assert 0.0 <= metrics["ndcg@5"] <= 1.0
    assert 0.0 <= metrics["ndcg@10"] <= 1.0
    assert content_hash(split.train) in metadata_path.read_text(encoding="utf-8")
