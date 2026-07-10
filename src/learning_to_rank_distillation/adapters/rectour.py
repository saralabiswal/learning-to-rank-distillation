"""RecTour adapter guarded by actual file schema validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from learning_to_rank_distillation.schema import RankingExample, coerce_feature_value


class RecTourDataNotFoundError(FileNotFoundError):
    """Raised when data/rectour has no readable data files."""


class RecTourSchemaError(ValueError):
    """Raised when RecTour files cannot be mapped without guessing."""


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}
KNOWN_FEATURE_COLUMNS = {
    "checkin_date",
    "checkout_date",
    "adult_count",
    "child_count",
    "length_of_stay",
    "booking_window",
    "destination_id",
    "point_of_sale",
    "geo_location_country",
    "is_mobile",
    "applied_filters",
    "sort_type",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "srch_room_count",
    "orig_destination_distance",
    "prop_country_id",
    "prop_id",
    "prop_brand_bool",
    "is_travel_ad",
    "is_free_cancellation",
    "is_drr",
    "star_rating",
    "review_rating",
    "review_count",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "price_bucket",
    "promotion_flag",
    "competitor_price_delta",
    "competitor_availability_rate",
    "price_rank",
    "star_rank",
    "location_score_rank",
    "price_diff_from_query_median",
    "historical_price_gap",
    "number_of_amenities",
    "free_parking",
    "parking",
    "gym",
    "swimming_pool",
    "washer_dryer",
    "spa_services",
    "free_airport_transportation",
}


@dataclass(frozen=True, slots=True)
class RecTourColumnMapping:
    """Explicit raw-column mapping into RankingExample fields."""

    query_id: str
    item_id: str
    label: str
    group_id: str | None = None
    is_unbiased: str | None = None
    position: str | None = None
    feature_columns: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class RecTourAdapter:
    """Load RecTour rows into the dataset-agnostic RankingExample schema.

    Implements M1, DR-1, DR-2, DR-3, and DR-4 boundaries. The adapter accepts an
    explicit mapping or canonical test-column names. It does not infer unknown
    Expedia fields from memory.
    """

    data_dir: Path = Path("data/rectour")
    column_mapping: RecTourColumnMapping | None = None

    def discover_files(self) -> list[Path]:
        data_dir = Path(self.data_dir)
        if not data_dir.exists():
            raise RecTourDataNotFoundError(f"{data_dir} does not exist")
        files = sorted(
            path
            for path in data_dir.iterdir()
            if path.is_file() and path.suffix in SUPPORTED_SUFFIXES
        )
        if not files:
            raise RecTourDataNotFoundError(
                f"{data_dir} contains no supported data files: {sorted(SUPPORTED_SUFFIXES)}"
            )
        return files

    def load(self, *, limit: int | None = None) -> list[RankingExample]:
        """Load all supported files and map rows into RankingExample objects."""

        import pandas as pd

        frames = [self._read_frame(path) for path in self.discover_files()]
        frame = pd.concat(frames, ignore_index=True)
        if limit is not None:
            frame = frame.head(limit)
        mapping = self.column_mapping or self._canonical_mapping(frame.columns)
        self._validate_mapping(frame.columns, mapping)
        return [self._row_to_example(row, mapping) for row in frame.to_dict(orient="records")]

    def _read_frame(self, path: Path) -> Any:
        import pandas as pd

        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".json":
            return pd.read_json(path)
        if path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        raise RecTourDataNotFoundError(f"unsupported RecTour file type: {path}")

    def _canonical_mapping(self, columns: Any) -> RecTourColumnMapping:
        column_set = set(columns)
        required = {"query_id", "item_id", "label"}
        if not required.issubset(column_set):
            raise RecTourSchemaError(
                "RecTour files need an explicit RecTourColumnMapping because canonical "
                f"columns {sorted(required)} were not all present. Observed columns: "
                f"{sorted(column_set)}"
            )

        features = tuple(sorted((column_set & KNOWN_FEATURE_COLUMNS) - {"position"}))
        return RecTourColumnMapping(
            query_id="query_id",
            item_id="item_id",
            group_id="group_id" if "group_id" in column_set else "item_id",
            label="label",
            is_unbiased="is_unbiased" if "is_unbiased" in column_set else None,
            position="position" if "position" in column_set else None,
            feature_columns=features,
        )

    def _validate_mapping(self, columns: Any, mapping: RecTourColumnMapping) -> None:
        column_set = set(columns)
        required = {mapping.query_id, mapping.item_id, mapping.label}
        if mapping.group_id:
            required.add(mapping.group_id)
        if mapping.is_unbiased:
            required.add(mapping.is_unbiased)
        if mapping.position:
            required.add(mapping.position)
        required.update(mapping.feature_columns)
        missing = sorted(required - column_set)
        if missing:
            raise RecTourSchemaError(f"mapped columns missing from RecTour data: {missing}")
        if "position" in mapping.feature_columns:
            raise RecTourSchemaError(
                "DR-1 violation: position cannot be used as a training feature"
            )
        if mapping.position and not mapping.is_unbiased:
            raise RecTourSchemaError("DR-1 requires is_unbiased when position is mapped")

    def _row_to_example(self, row: dict[str, Any], mapping: RecTourColumnMapping) -> RankingExample:
        is_unbiased = bool(row[mapping.is_unbiased]) if mapping.is_unbiased else False
        position = int(row[mapping.position]) if mapping.position and is_unbiased else None
        group_column = mapping.group_id or mapping.item_id
        return RankingExample(
            query_id=str(row[mapping.query_id]),
            item_id=str(row[mapping.item_id]),
            group_id=str(row[group_column]),
            label=float(row[mapping.label]),
            is_unbiased=is_unbiased,
            position=position,
            features={
                column: coerce_feature_value(row[column])
                for column in mapping.feature_columns
                if column not in {mapping.position}
            },
        )
