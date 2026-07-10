"""MovieLens adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learning_to_rank_distillation.schema import RankingExample, coerce_feature_value


class MovieLensDataNotFoundError(FileNotFoundError):
    """Raised when data/movielens has no readable MovieLens files."""


class MovieLensSchemaError(ValueError):
    """Raised when MovieLens files cannot be mapped safely."""


RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"
RATING_COLUMNS = {"userId", "movieId", "rating"}
MOVIE_COLUMNS = {"movieId", "title", "genres"}


@dataclass(slots=True)
class MovieLensAdapter:
    """Load MovieLens ratings into the dataset-agnostic RankingExample schema."""

    data_dir: Path = Path("data/movielens")
    min_rating: float | None = None

    def discover_files(self) -> list[Path]:
        data_dir = Path(self.data_dir)
        if not data_dir.exists():
            raise MovieLensDataNotFoundError(f"{data_dir} does not exist")
        ratings_path = data_dir / RATINGS_FILE
        if not ratings_path.exists():
            raise MovieLensDataNotFoundError(f"{data_dir} must contain {RATINGS_FILE}")
        files = [ratings_path]
        movies_path = data_dir / MOVIES_FILE
        if movies_path.exists():
            files.append(movies_path)
        return files

    def load(self, *, limit: int | None = None) -> list[RankingExample]:
        """Load MovieLens rows and optional movie metadata."""

        ratings = self._read_ratings()
        if self.min_rating is not None:
            ratings = ratings[ratings["rating"] >= self.min_rating]
        if limit is not None:
            ratings = ratings.head(limit)
        if ratings.empty:
            raise MovieLensSchemaError("MovieLens filters produced no rows")

        frame = self._merge_movie_metadata(ratings)
        return [self._row_to_example(row) for row in frame.to_dict(orient="records")]

    def _read_ratings(self) -> Any:
        import pandas as pd

        self.discover_files()
        frame = pd.read_csv(Path(self.data_dir) / RATINGS_FILE)
        missing = sorted(RATING_COLUMNS - set(frame))
        if missing:
            raise MovieLensSchemaError(f"MovieLens ratings missing required columns: {missing}")
        return frame

    def _merge_movie_metadata(self, ratings: Any) -> Any:
        import pandas as pd

        movies_path = Path(self.data_dir) / MOVIES_FILE
        if not movies_path.exists():
            return ratings
        movies = pd.read_csv(movies_path)
        missing = sorted({"movieId"} - set(movies))
        if missing:
            raise MovieLensSchemaError(f"MovieLens movies missing required columns: {missing}")
        columns = [column for column in MOVIE_COLUMNS if column in movies.columns]
        movies = movies[columns].drop_duplicates(["movieId"])
        return ratings.merge(movies, how="left", on="movieId")

    def _row_to_example(self, row: dict[str, Any]) -> RankingExample:
        movie_id = str(row["movieId"])
        genres = _genres(row.get("genres"))
        primary_genre = genres[0] if genres else movie_id
        return RankingExample(
            query_id=f"user-{row['userId']}",
            item_id=f"movie-{movie_id}",
            group_id=primary_genre,
            label=float(row["rating"]),
            is_unbiased=False,
            position=None,
            features=_features_from_row(row, genres),
        )


def _features_from_row(row: dict[str, Any], genres: tuple[str, ...]) -> dict[str, Any]:
    title = _clean_text(row.get("title"))
    release_year = _release_year(title)
    features: dict[str, Any] = {
        "user_id": str(row["userId"]),
        "movie_id": str(row["movieId"]),
        "primary_genre": genres[0] if genres else None,
        "genre_count": len(genres),
        "release_year": release_year,
        "title_char_count": len(title),
        "title_token_count": len(_tokens(title)),
    }
    for genre in genres:
        features[f"genre_{_feature_name(genre)}"] = True
    return {name: coerce_feature_value(value) for name, value in features.items()}


def _genres(value: Any) -> tuple[str, ...]:
    text = _clean_text(value)
    if not text or text == "(no genres listed)":
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _release_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", title)
    return None if match is None else int(match.group(1))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower()))


def _feature_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    return str(value).strip()
