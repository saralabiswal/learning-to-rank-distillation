import numpy as np

from learning_to_rank_distillation.features import FeatureVectorizer
from tests.fixtures.synthetic_ranking_data import make_synthetic_ranking_data


def test_feature_vectorizer_handles_numeric_and_categorical_features() -> None:
    examples = make_synthetic_ranking_data(num_queries=3, items_per_query=2)
    vectorizer = FeatureVectorizer()

    features = vectorizer.fit_transform(examples)
    transformed = vectorizer.transform(examples[:2])

    assert features.shape[0] == len(examples)
    assert features.shape[1] == vectorizer.output_dim()
    assert transformed.shape == (2, vectorizer.output_dim())
    assert np.isfinite(features).all()
