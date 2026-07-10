"""Feature vectorization shared across models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from learning_to_rank_distillation.schema import (
    FeatureValue,
    RankingExample,
    group_by_query,
    sorted_feature_names,
)


@dataclass(slots=True)
class FeatureVectorizer:
    """Convert adapter-defined feature dicts into model-ready numeric arrays."""

    numeric_features: list[str] = field(default_factory=list)
    categorical_values: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, examples: list[RankingExample]) -> FeatureVectorizer:
        names = sorted_feature_names(examples)
        numeric: list[str] = []
        categorical: dict[str, set[str]] = {}
        for name in names:
            values = [example.features.get(name) for example in examples]
            observed = [value for value in values if value is not None]
            if all(isinstance(value, bool | int | float) for value in observed):
                numeric.append(name)
            else:
                categorical[name] = {str(value) for value in observed}

        self.numeric_features = numeric
        self.categorical_values = {
            name: sorted(values) for name, values in sorted(categorical.items())
        }
        return self

    def transform(self, examples: list[RankingExample]) -> np.ndarray:
        rows = [self._transform_one(example.features) for example in examples]
        return np.asarray(rows, dtype=np.float32)

    def fit_transform(self, examples: list[RankingExample]) -> np.ndarray:
        return self.fit(examples).transform(examples)

    def output_dim(self) -> int:
        return len(self.numeric_features) + sum(
            len(values) for values in self.categorical_values.values()
        )

    def _transform_one(self, features: dict[str, FeatureValue]) -> list[float]:
        row: list[float] = []
        for name in self.numeric_features:
            value = features.get(name)
            row.append(0.0 if value is None else float(value))
        for name, values in self.categorical_values.items():
            value = features.get(name)
            text = None if value is None else str(value)
            row.extend(1.0 if text == category else 0.0 for category in values)
        return row


@dataclass(slots=True)
class TwoTowerVectorizer:
    """Vectorize RankingExample features into query-side and item-side arrays.

    Implements FR-2.1 without dataset-specific feature names.
    """

    query_feature_names: list[str] = field(default_factory=list)
    item_feature_names: list[str] = field(default_factory=list)
    query_vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    item_vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)

    def fit(self, examples: list[RankingExample]) -> TwoTowerVectorizer:
        self.query_feature_names = infer_query_feature_names(examples)
        all_features = set(sorted_feature_names(examples))
        self.item_feature_names = sorted(all_features - set(self.query_feature_names))
        self.query_vectorizer.fit(_examples_with_features(examples, self.query_feature_names))
        self.item_vectorizer.fit(_examples_with_features(examples, self.item_feature_names))
        return self

    def transform_query(self, examples: list[RankingExample]) -> np.ndarray:
        return self.query_vectorizer.transform(
            _examples_with_features(examples, self.query_feature_names)
        )

    def transform_item(self, examples: list[RankingExample]) -> np.ndarray:
        return self.item_vectorizer.transform(
            _examples_with_features(examples, self.item_feature_names)
        )

    def query_dim(self) -> int:
        return self.query_vectorizer.output_dim()

    def item_dim(self) -> int:
        return self.item_vectorizer.output_dim()


def infer_query_feature_names(examples: list[RankingExample]) -> list[str]:
    """Infer query-side features as values that are constant within each query."""

    names = sorted_feature_names(examples)
    grouped = group_by_query(examples)
    query_features: list[str] = []
    for name in names:
        if all(_feature_is_constant(query_examples, name) for query_examples in grouped.values()):
            query_features.append(name)
    return query_features


def _feature_is_constant(examples: list[RankingExample], name: str) -> bool:
    values = {example.features.get(name) for example in examples}
    return len(values) <= 1


def _examples_with_features(
    examples: list[RankingExample],
    feature_names: list[str],
) -> list[RankingExample]:
    wanted = set(feature_names)
    return [
        RankingExample(
            query_id=example.query_id,
            item_id=example.item_id,
            group_id=example.group_id,
            label=example.label,
            is_unbiased=example.is_unbiased,
            position=example.position,
            features={name: value for name, value in example.features.items() if name in wanted},
        )
        for example in examples
    ]
