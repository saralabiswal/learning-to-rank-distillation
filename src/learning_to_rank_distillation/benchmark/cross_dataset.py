"""Cross-dataset benchmark comparison.

Author: Sarala Biswal
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from learning_to_rank_distillation.benchmark.run_all import run_benchmark
from learning_to_rank_distillation.datasets import DatasetName


@dataclass(frozen=True, slots=True)
class CrossDatasetResult:
    dataset: str
    status: str
    artifact_dir: str
    rows: list[dict[str, object]]
    error: str | None = None


def run_cross_dataset_benchmark(
    *,
    datasets: Sequence[DatasetName] = ("synthetic", "esci", "rectour", "movielens"),
    data_dirs: Mapping[str, Path] | None = None,
    output_dir: Path = Path("artifacts/cross_dataset"),
    num_queries: int = 12,
    items_per_query: int = 4,
    student_epochs: int = 1,
    seed: int = 13,
    limit: int | None = None,
) -> list[CrossDatasetResult]:
    """Run the benchmark across available datasets and skip missing raw-data directories."""

    output_dir.mkdir(parents=True, exist_ok=True)
    configured_dirs = data_dirs or {}
    results: list[CrossDatasetResult] = []
    for dataset in datasets:
        dataset_output_dir = output_dir / dataset
        try:
            rows = run_benchmark(
                dataset=dataset,
                data_dir=configured_dirs.get(dataset),
                num_queries=num_queries,
                items_per_query=items_per_query,
                student_epochs=student_epochs,
                student_embedding_dims=(8,),
                output_dir=dataset_output_dir,
                seed=seed,
                limit=limit,
            )
        except FileNotFoundError as exc:
            results.append(
                CrossDatasetResult(
                    dataset=dataset,
                    status="skipped",
                    artifact_dir=str(dataset_output_dir),
                    rows=[],
                    error=str(exc),
                )
            )
        else:
            results.append(
                CrossDatasetResult(
                    dataset=dataset,
                    status="completed",
                    artifact_dir=str(dataset_output_dir),
                    rows=[asdict(row) for row in rows],
                )
            )

    (output_dir / "cross_dataset_benchmark.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return results


def main(argv: Sequence[str] | None = None) -> None:
    # Cross-dataset runs validate the adapter boundary: each dataset should
    # produce the shared contract without changing model or benchmark code.
    parser = argparse.ArgumentParser(description="Run cross-dataset benchmark comparison.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["synthetic", "esci", "rectour", "movielens"],
        default=["synthetic", "esci", "rectour", "movielens"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cross_dataset"))
    parser.add_argument("--num-queries", type=int, default=12)
    parser.add_argument("--items-per-query", type=int, default=4)
    parser.add_argument("--student-epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    # Print JSON because this command is mostly used as regression evidence and
    # can be consumed directly by docs, dashboards, or CI diagnostics.
    results = run_cross_dataset_benchmark(
        datasets=tuple(args.datasets),
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        items_per_query=args.items_per_query,
        student_epochs=args.student_epochs,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
