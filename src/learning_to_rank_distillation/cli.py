"""Command line interface for the ltrd package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from learning_to_rank_distillation.adapters.synthetic import write_synthetic_rectour_csv
from learning_to_rank_distillation.benchmark.run_all import (
    format_markdown_table,
    run_benchmark,
)
from learning_to_rank_distillation.datasets import DatasetConfig, load_ranking_examples
from learning_to_rank_distillation.models.teacher import LightGBMLambdaMARTTeacher

DATASET_CHOICES = ("synthetic", "esci", "rectour")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ltrd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run the v1.0 benchmark")
    benchmark.add_argument("--dataset", choices=DATASET_CHOICES, default="synthetic")
    benchmark.add_argument("--data-dir", type=Path, default=None)
    benchmark.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    benchmark.add_argument("--num-queries", type=int, default=36)
    benchmark.add_argument("--items-per-query", type=int, default=8)
    benchmark.add_argument("--student-epochs", type=int, default=4)
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument("--split", default="train")
    benchmark.add_argument("--seed", type=int, default=13)

    train_teacher = subparsers.add_parser("train-teacher", help="Train and save a teacher model")
    train_teacher.add_argument("--dataset", choices=DATASET_CHOICES, default="synthetic")
    train_teacher.add_argument("--data-dir", type=Path, default=None)
    train_teacher.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    train_teacher.add_argument("--num-queries", type=int, default=36)
    train_teacher.add_argument("--items-per-query", type=int, default=8)
    train_teacher.add_argument("--limit", type=int, default=None)
    train_teacher.add_argument("--split", default="train")
    train_teacher.add_argument("--seed", type=int, default=13)

    generate_synthetic = subparsers.add_parser(
        "generate-synthetic-rectour",
        help="Write a RecTour-like synthetic CSV for inspection or demos",
    )
    generate_synthetic.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/synthetic/rectour_like.csv"),
    )
    generate_synthetic.add_argument("--num-queries", type=int, default=24)
    generate_synthetic.add_argument("--items-per-query", type=int, default=8)
    generate_synthetic.add_argument("--seed", type=int, default=13)

    args = parser.parse_args(argv)
    if args.command == "benchmark":
        rows = run_benchmark(
            dataset=args.dataset,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_queries=args.num_queries,
            items_per_query=args.items_per_query,
            student_epochs=args.student_epochs,
            seed=args.seed,
            limit=args.limit,
            split=args.split,
        )
        print(format_markdown_table(rows))
    elif args.command == "train-teacher":
        examples = load_ranking_examples(
            DatasetConfig(
                name=args.dataset,
                data_dir=args.data_dir,
                num_queries=args.num_queries,
                items_per_query=args.items_per_query,
                seed=args.seed,
                limit=args.limit,
                split=args.split,
            )
        )
        teacher = LightGBMLambdaMARTTeacher(random_state=args.seed).fit(examples)
        metadata_path = teacher.save(args.output_dir, examples)
        print(metadata_path)
    elif args.command == "generate-synthetic-rectour":
        output_path = write_synthetic_rectour_csv(
            args.output_path,
            num_queries=args.num_queries,
            items_per_query=args.items_per_query,
            seed=args.seed,
        )
        print(output_path)


if __name__ == "__main__":
    main()
