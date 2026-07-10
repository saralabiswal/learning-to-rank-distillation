"""Amazon ESCI Shopping Queries Dataset adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learning_to_rank_distillation.schema import RankingExample, coerce_feature_value


class ESCIDataNotFoundError(FileNotFoundError):
    """Raised when data/esci has no readable ESCI files."""


class ESCISchemaError(ValueError):
    """Raised when ESCI files cannot be mapped safely."""


SUPPORTED_SUFFIXES = {".csv", ".parquet"}
EXAMPLES_STEM = "shopping_queries_dataset_examples"
PRODUCTS_STEM = "shopping_queries_dataset_products"
SOURCES_STEM = "shopping_queries_dataset_sources"
LABELS = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}
EXAMPLE_COLUMNS = {
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "large_version",
    "split",
}
PRODUCT_COLUMNS = {
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
}


@dataclass(slots=True)
class ESCIAdapter:
    """Load Amazon ESCI rows into the dataset-agnostic RankingExample schema."""

    data_dir: Path = Path("data/esci")
    split: str | None = "train"
    small_version_only: bool = True
    locales: tuple[str, ...] | None = None

    def discover_files(self) -> list[Path]:
        data_dir = Path(self.data_dir)
        if not data_dir.exists():
            raise ESCIDataNotFoundError(f"{data_dir} does not exist")
        files = sorted(
            path
            for path in data_dir.iterdir()
            if path.is_file() and path.suffix in SUPPORTED_SUFFIXES
        )
        if not files:
            raise ESCIDataNotFoundError(
                f"{data_dir} contains no supported data files: {sorted(SUPPORTED_SUFFIXES)}"
            )
        if not self._find_file(EXAMPLES_STEM):
            raise ESCIDataNotFoundError(
                f"{data_dir} must contain {EXAMPLES_STEM}.csv or {EXAMPLES_STEM}.parquet"
            )
        return files

    def load(self, *, limit: int | None = None) -> list[RankingExample]:
        """Load ESCI examples and optional product/source metadata."""

        examples = self._read_examples()
        examples = self._filter_examples(examples)
        if limit is not None:
            examples = examples.head(limit)

        frame = self._merge_optional_metadata(examples)
        return [self._row_to_example(row) for row in frame.to_dict(orient="records")]

    def _find_file(self, stem: str) -> Path | None:
        data_dir = Path(self.data_dir)
        for suffix in SUPPORTED_SUFFIXES:
            path = data_dir / f"{stem}{suffix}"
            if path.exists():
                return path
        return None

    def _read_examples(self) -> Any:
        path = self._find_file(EXAMPLES_STEM)
        if path is None:
            self.discover_files()
            raise ESCIDataNotFoundError(f"missing {EXAMPLES_STEM}")
        frame = self._read_frame(path)
        missing = sorted({"query_id", "product_id", "product_locale", "esci_label"} - set(frame))
        if missing:
            raise ESCISchemaError(f"ESCI examples missing required columns: {missing}")
        return frame

    def _filter_examples(self, frame: Any) -> Any:
        if self.small_version_only and "small_version" in frame:
            frame = frame[frame["small_version"] == 1]
        if self.split and "split" in frame:
            frame = frame[frame["split"] == self.split]
        if self.locales:
            frame = frame[frame["product_locale"].isin(self.locales)]
        if frame.empty:
            raise ESCISchemaError("ESCI filters produced no rows")
        return frame

    def _merge_optional_metadata(self, examples: Any) -> Any:
        products_path = self._find_file(PRODUCTS_STEM)
        if products_path:
            products = self._read_frame(products_path, columns=PRODUCT_COLUMNS)
            products = products.drop_duplicates(["product_locale", "product_id"])
            examples = examples.merge(
                products,
                how="left",
                on=["product_locale", "product_id"],
            )

        sources_path = self._find_file(SOURCES_STEM)
        if sources_path:
            sources = self._read_frame(sources_path)
            if {"query_id", "source"}.issubset(sources.columns):
                sources = sources.drop_duplicates(["query_id"])
                examples = examples.merge(
                    sources[["query_id", "source"]], how="left", on="query_id"
                )

        return examples

    def _read_frame(self, path: Path, *, columns: set[str] | None = None) -> Any:
        import pandas as pd

        if path.suffix == ".csv":
            if columns is None:
                return pd.read_csv(path)
            header = pd.read_csv(path, nrows=0).columns
            usecols = [column for column in columns if column in header]
            return pd.read_csv(path, usecols=usecols)
        if path.suffix == ".parquet":
            if columns is None:
                return pd.read_parquet(path)
            try:
                return pd.read_parquet(path, columns=list(columns))
            except ValueError:
                frame = pd.read_parquet(path)
                return frame[[column for column in columns if column in frame.columns]]
        raise ESCIDataNotFoundError(f"unsupported ESCI file type: {path}")

    def _row_to_example(self, row: dict[str, Any]) -> RankingExample:
        label = _label_to_relevance(row["esci_label"])
        product_id = str(row["product_id"])
        return RankingExample(
            query_id=str(row["query_id"]),
            item_id=product_id,
            group_id=str(_clean_value(row.get("product_brand")) or product_id),
            label=label,
            is_unbiased=False,
            position=None,
            features=_features_from_row(row),
        )


def _label_to_relevance(label: Any) -> float:
    text = str(label).strip().upper()
    if text not in LABELS:
        raise ESCISchemaError(f"unsupported ESCI label: {label!r}")
    return LABELS[text]


def _features_from_row(row: dict[str, Any]) -> dict[str, Any]:
    query = _clean_text(row.get("query"))
    title = _clean_text(row.get("product_title"))
    description = _clean_text(row.get("product_description"))
    bullet = _clean_text(row.get("product_bullet_point"))
    brand = _clean_text(row.get("product_brand"))
    color = _clean_text(row.get("product_color"))

    query_tokens = _tokens(query)
    title_tokens = _tokens(title)
    description_tokens = _tokens(description)
    bullet_tokens = _tokens(bullet)
    brand_tokens = _tokens(brand)

    features = {
        "product_locale": _clean_value(row.get("product_locale")),
        "source": _clean_value(row.get("source")),
        "product_brand": _clean_value(row.get("product_brand")),
        "product_color": _clean_value(row.get("product_color")),
        "query_char_count": len(query),
        "query_token_count": len(query_tokens),
        "title_char_count": len(title),
        "title_token_count": len(title_tokens),
        "description_token_count": len(description_tokens),
        "bullet_token_count": len(bullet_tokens),
        "query_title_token_overlap": _overlap(query_tokens, title_tokens),
        "query_description_token_overlap": _overlap(query_tokens, description_tokens),
        "query_bullet_token_overlap": _overlap(query_tokens, bullet_tokens),
        "query_brand_token_overlap": _overlap(query_tokens, brand_tokens),
        "has_product_description": bool(description),
        "has_product_bullet_point": bool(bullet),
        "has_product_brand": bool(brand),
        "has_product_color": bool(color),
    }
    return {name: coerce_feature_value(value) for name, value in features.items()}


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        if value != value:
            return None
    except TypeError:
        return value
    return value


def _clean_text(value: Any) -> str:
    value = _clean_value(value)
    return "" if value is None else str(value).strip().lower()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower()))


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
