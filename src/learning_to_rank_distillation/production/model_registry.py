"""Filesystem model registry for bundle versions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    version: str
    bundle_path: str
    stage: str
    metrics: dict[str, float]
    created_at: str


class FileModelRegistry:
    """Small local model registry for production-shaped bundle promotion."""

    def __init__(self, registry_path: Path = Path("artifacts/model_registry.json")) -> None:
        self.registry_path = registry_path

    def register(
        self,
        *,
        version: str,
        bundle_path: Path,
        metrics: dict[str, float] | None = None,
        stage: str = "candidate",
    ) -> RegistryEntry:
        records = self._read()
        entry = RegistryEntry(
            version=version,
            bundle_path=str(bundle_path),
            stage=stage,
            metrics=metrics or {},
            created_at=datetime.now(UTC).isoformat(),
        )
        records[version] = {
            "version": entry.version,
            "bundle_path": entry.bundle_path,
            "stage": entry.stage,
            "metrics": entry.metrics,
            "created_at": entry.created_at,
        }
        self._write(records)
        return entry

    def promote(self, version: str, *, stage: str = "production") -> RegistryEntry:
        records = self._read()
        if version not in records:
            raise KeyError(f"unknown model version: {version}")
        records[version]["stage"] = stage
        records[version]["promoted_at"] = datetime.now(UTC).isoformat()
        self._write(records)
        record = records[version]
        return RegistryEntry(
            version=str(record["version"]),
            bundle_path=str(record["bundle_path"]),
            stage=str(record["stage"]),
            metrics=dict(record.get("metrics", {})),
            created_at=str(record["created_at"]),
        )

    def latest(self, *, stage: str = "production") -> RegistryEntry | None:
        records = [record for record in self._read().values() if record.get("stage") == stage]
        if not records:
            return None
        record = max(records, key=lambda value: str(value.get("promoted_at", value["created_at"])))
        return RegistryEntry(
            version=str(record["version"]),
            bundle_path=str(record["bundle_path"]),
            stage=str(record["stage"]),
            metrics=dict(record.get("metrics", {})),
            created_at=str(record["created_at"]),
        )

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write(self, records: dict[str, dict[str, Any]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
