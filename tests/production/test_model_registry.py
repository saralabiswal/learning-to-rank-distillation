from pathlib import Path

from learning_to_rank_distillation.production.model_registry import FileModelRegistry


def test_file_model_registry_registers_and_promotes_version(tmp_path: Path) -> None:
    registry = FileModelRegistry(tmp_path / "registry.json")

    entry = registry.register(
        version="model-1",
        bundle_path=tmp_path / "bundle",
        metrics={"ndcg_at_5": 0.8},
    )
    promoted = registry.promote(entry.version)
    latest = registry.latest()

    assert entry.stage == "candidate"
    assert promoted.stage == "production"
    assert latest is not None
    assert latest.version == "model-1"
    assert latest.metrics == {"ndcg_at_5": 0.8}
