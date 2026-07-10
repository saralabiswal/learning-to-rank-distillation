"""Distillation-method ablation benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from learning_to_rank_distillation.benchmark.metrics import ndcg_at_k
from learning_to_rank_distillation.data import split_by_query
from learning_to_rank_distillation.datasets import DatasetConfig, DatasetName, load_ranking_examples
from learning_to_rank_distillation.distillation.feature_based import train_feature_based_student
from learning_to_rank_distillation.distillation.no_kd_baseline import train_no_kd_student
from learning_to_rank_distillation.distillation.relation_based import train_relation_based_student
from learning_to_rank_distillation.distillation.response_based import train_response_based_student
from learning_to_rank_distillation.models.student import StudentConfig, TwoTowerStudent
from learning_to_rank_distillation.models.transformer_teacher import TransformerRankerTeacher
from learning_to_rank_distillation.schema import RankingExample


@dataclass(frozen=True, slots=True)
class DistillationAblationRow:
    model: str
    ndcg_at_5: float
    ndcg_at_10: float
    final_loss: float | None


def run_distillation_ablation(
    *,
    dataset: DatasetName = "synthetic",
    data_dir: Path | None = None,
    num_queries: int = 24,
    items_per_query: int = 6,
    output_dir: Path = Path("artifacts"),
    seed: int = 13,
    limit: int | None = None,
    split: str | None = "train",
    student_epochs: int = 4,
    teacher_epochs: int = 4,
    embedding_dim: int = 16,
) -> list[DistillationAblationRow]:
    """Compare no-KD, response-KD, feature-KD, and relation-KD head-to-head."""

    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_ranking_examples(
        DatasetConfig(
            name=dataset,
            data_dir=data_dir,
            num_queries=num_queries,
            items_per_query=items_per_query,
            seed=seed,
            limit=limit,
            split=split,
        )
    )
    query_split = split_by_query(examples, seed=seed)
    teacher = TransformerRankerTeacher(random_state=seed, epochs=teacher_epochs).fit(
        query_split.train
    )
    teacher_train_scores = teacher.predict(query_split.train)
    teacher_train_representations = teacher.item_representations(query_split.train)
    teacher_test_scores = teacher.predict(query_split.test)

    rows = [
        DistillationAblationRow(
            model="teacher-transformer",
            ndcg_at_5=ndcg_at_k(query_split.test, teacher_test_scores, k=5),
            ndcg_at_10=ndcg_at_k(query_split.test, teacher_test_scores, k=10),
            final_loss=None,
        )
    ]

    student_config = StudentConfig(embedding_dim=embedding_dim, random_state=seed)
    no_kd_student, no_kd_history = train_no_kd_student(
        query_split.train,
        config=student_config,
        epochs=student_epochs,
    )
    rows.append(
        _student_row(
            f"student-no-kd-d{embedding_dim}",
            no_kd_student,
            query_split.test,
            no_kd_history.final_loss,
        )
    )

    response_student, response_history = train_response_based_student(
        query_split.train,
        teacher_train_scores,
        config=student_config,
        epochs=student_epochs,
    )
    rows.append(
        _student_row(
            f"student-response-kd-d{embedding_dim}",
            response_student,
            query_split.test,
            response_history.final_loss,
        )
    )

    feature_student, feature_history = train_feature_based_student(
        query_split.train,
        teacher_train_representations,
        config=student_config,
        epochs=student_epochs,
    )
    rows.append(
        _student_row(
            f"student-feature-kd-d{embedding_dim}",
            feature_student,
            query_split.test,
            feature_history.final_loss,
        )
    )

    relation_student, relation_history = train_relation_based_student(
        query_split.train,
        teacher_train_scores,
        config=student_config,
        epochs=student_epochs,
    )
    rows.append(
        _student_row(
            f"student-relation-kd-d{embedding_dim}",
            relation_student,
            query_split.test,
            relation_history.final_loss,
        )
    )

    _write_outputs(rows, output_dir)
    return rows


def _student_row(
    model_name: str,
    student: TwoTowerStudent,
    examples: list[RankingExample],
    final_loss: float,
) -> DistillationAblationRow:
    scores = student.predict(examples)
    return DistillationAblationRow(
        model=model_name,
        ndcg_at_5=ndcg_at_k(examples, scores, k=5),
        ndcg_at_10=ndcg_at_k(examples, scores, k=10),
        final_loss=final_loss,
    )


def _write_outputs(rows: list[DistillationAblationRow], output_dir: Path) -> None:
    (output_dir / "distillation_ablation.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def format_markdown_table(rows: list[DistillationAblationRow]) -> str:
    header = "| model | NDCG@5 | NDCG@10 | final loss |\n|---|---:|---:|---:|"
    lines = [header]
    for row in rows:
        final_loss = "-" if row.final_loss is None else f"{row.final_loss:.4f}"
        lines.append(f"| {row.model} | {row.ndcg_at_5:.4f} | {row.ndcg_at_10:.4f} | {final_loss} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the distillation-method ablation.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "esci", "rectour", "movielens"],
        default="synthetic",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--num-queries", type=int, default=24)
    parser.add_argument("--items-per-query", type=int, default=6)
    parser.add_argument("--student-epochs", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    rows = run_distillation_ablation(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        items_per_query=args.items_per_query,
        student_epochs=args.student_epochs,
        teacher_epochs=args.teacher_epochs,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        limit=args.limit,
        split=args.split,
    )
    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()
