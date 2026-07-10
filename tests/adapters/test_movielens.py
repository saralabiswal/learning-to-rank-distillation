from pathlib import Path

import pytest

from learning_to_rank_distillation.adapters.movielens import (
    MovieLensAdapter,
    MovieLensDataNotFoundError,
    MovieLensSchemaError,
)


def test_movielens_adapter_reports_missing_data(tmp_path: Path) -> None:
    with pytest.raises(MovieLensDataNotFoundError):
        MovieLensAdapter(tmp_path / "missing").discover_files()


def test_movielens_adapter_validates_required_columns(tmp_path: Path) -> None:
    data_dir = tmp_path / "movielens"
    data_dir.mkdir()
    (data_dir / "ratings.csv").write_text("userId,movieId\n1,10\n", encoding="utf-8")

    with pytest.raises(MovieLensSchemaError, match="rating"):
        MovieLensAdapter(data_dir).load()


def test_movielens_adapter_loads_ratings_and_movie_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "movielens"
    _write_sample_movielens_data(data_dir)

    examples = MovieLensAdapter(data_dir).load()

    assert [example.query_id for example in examples] == ["user-1", "user-1", "user-2"]
    assert [example.item_id for example in examples] == ["movie-10", "movie-20", "movie-10"]
    assert [example.label for example in examples] == [4.5, 2.0, 5.0]
    assert examples[0].group_id == "Adventure"
    assert examples[0].features["primary_genre"] == "Adventure"
    assert examples[0].features["release_year"] == 1995
    assert examples[0].features["genre_adventure"] is True
    assert examples[1].features["genre_count"] == 1


def _write_sample_movielens_data(data_dir: Path) -> None:
    data_dir.mkdir()
    (data_dir / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,4.5,964982703\n"
        "1,20,2.0,964982224\n"
        "2,10,5.0,964981247\n",
        encoding="utf-8",
    )
    (data_dir / "movies.csv").write_text(
        "movieId,title,genres\n"
        "10,Toy Story (1995),Adventure|Animation|Children\n"
        "20,Sabrina (1995),Comedy\n",
        encoding="utf-8",
    )
