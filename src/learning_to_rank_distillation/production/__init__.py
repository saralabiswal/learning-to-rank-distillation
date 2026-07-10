"""Production-shaped serving and artifact utilities."""

from learning_to_rank_distillation.production.bundle import (
    StudentBundle,
    load_student_bundle,
    save_student_bundle,
)
from learning_to_rank_distillation.production.index_lifecycle import (
    BundleBuildResult,
    build_student_bundle_version,
    publish_bundle_version,
    resolve_published_bundle,
    validate_student_bundle,
)
from learning_to_rank_distillation.production.model_registry import FileModelRegistry, RegistryEntry
from learning_to_rank_distillation.production.tracking import (
    benchmark_metrics_from_rows,
    log_experiment_run,
)

__all__ = [
    "BundleBuildResult",
    "FileModelRegistry",
    "RegistryEntry",
    "StudentBundle",
    "build_student_bundle_version",
    "benchmark_metrics_from_rows",
    "load_student_bundle",
    "log_experiment_run",
    "publish_bundle_version",
    "resolve_published_bundle",
    "save_student_bundle",
    "validate_student_bundle",
]
