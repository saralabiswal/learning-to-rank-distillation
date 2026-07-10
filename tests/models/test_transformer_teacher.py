from pathlib import Path

import numpy as np

from learning_to_rank_distillation.models.transformer_teacher import TransformerRankerTeacher
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_transformer_teacher_trains_predicts_and_exposes_representations(tmp_path: Path) -> None:
    examples = make_synthetic_ranking_data(num_queries=5, items_per_query=4, seed=21)
    teacher = TransformerRankerTeacher(
        random_state=21,
        hidden_dim=16,
        num_heads=4,
        epochs=2,
    ).fit(examples)

    scores = teacher.predict(examples)
    representations = teacher.item_representations(examples)
    metadata_path = teacher.save(tmp_path, examples)

    assert scores.shape == (len(examples),)
    assert np.isfinite(scores).all()
    assert representations.shape == (len(examples), 16)
    assert np.isfinite(representations).all()
    assert metadata_path.exists()
    assert metadata_path.with_suffix(".pt").exists()
