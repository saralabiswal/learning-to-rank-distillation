"""Shared dataset contract for ranking examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FeatureValue = bool | int | float | str | None


@dataclass(frozen=True, slots=True)
class RankingExample:
    """The dataset-agnostic row contract used by every downstream component.

    Implements REQUIREMENTS.md Section 1.
    """

    query_id: str
    item_id: str
    features: dict[str, FeatureValue]
    label: float
    group_id: str
    is_unbiased: bool
    position: int | None

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        if not self.item_id:
            raise ValueError("item_id must be non-empty")
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if not isinstance(self.features, dict):
            raise TypeError("features must be a dictionary")
        if self.position is not None and self.position < 1:
            raise ValueError("position must be 1-indexed when provided")
        if "position" in self.features:
            raise ValueError("position must not be included as a training feature")


def validate_examples(examples: list[RankingExample]) -> None:
    """Validate a list of examples before training or evaluation.

    Implements DR-1 and DR-2 validation support.
    """

    if not examples:
        raise ValueError("examples must be non-empty")
    for example in examples:
        if not isinstance(example, RankingExample):
            raise TypeError(f"expected RankingExample, got {type(example)!r}")


def group_by_query(examples: list[RankingExample]) -> dict[str, list[RankingExample]]:
    """Return examples grouped by query_id in first-seen query order."""

    validate_examples(examples)
    grouped: dict[str, list[RankingExample]] = {}
    for example in examples:
        grouped.setdefault(example.query_id, []).append(example)
    return grouped


def sorted_feature_names(examples: list[RankingExample]) -> list[str]:
    """Return a stable union of feature names across examples."""

    validate_examples(examples)
    names: set[str] = set()
    for example in examples:
        names.update(example.features)
    return sorted(names)


def coerce_feature_value(value: Any) -> FeatureValue:
    """Normalize raw feature values to the supported schema value types."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)
