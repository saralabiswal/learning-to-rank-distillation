"""FastAPI serving endpoint for student bundles."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from learning_to_rank_distillation.production.bundle import StudentBundle, load_student_bundle
from learning_to_rank_distillation.schema import FeatureValue, RankingExample


class RankRequest(BaseModel):
    query_features: dict[str, FeatureValue] = Field(default_factory=dict)
    k: int = Field(default=10, ge=1, le=100)


class RankResult(BaseModel):
    item_id: str
    score: float


def create_app(
    *,
    bundle_path: Path | None = None,
    bundle: StudentBundle | None = None,
) -> FastAPI:
    """Create a FastAPI app around a loaded student bundle."""

    app = FastAPI(title="learning-to-rank-distillation serving")
    registry = CollectorRegistry()
    metrics = _ServingMetrics(registry)
    loaded_bundle = bundle or _load_bundle_from_path(bundle_path)
    item_index = loaded_bundle.item_index() if loaded_bundle else None
    if loaded_bundle:
        metrics.item_count.set(len(loaded_bundle.item_ids))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if loaded_bundle else "unconfigured",
            "item_count": 0 if loaded_bundle is None else len(loaded_bundle.item_ids),
            "bundle_path": None if loaded_bundle is None else str(loaded_bundle.path),
        }

    @app.post("/rank")
    def rank(request: RankRequest) -> dict[str, Any]:
        if loaded_bundle is None or item_index is None:
            metrics.errors.labels(kind="bundle_missing").inc()
            raise HTTPException(status_code=503, detail="student bundle is not configured")

        started = time.perf_counter()
        try:
            query_example = _query_example(request.query_features)
            with torch.no_grad():
                query_embeddings = loaded_bundle.student.query_embeddings([query_example])
            results = item_index.search(query_embeddings, k=request.k)[0]
        except Exception as exc:
            metrics.errors.labels(kind=type(exc).__name__).inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            metrics.latency.observe(time.perf_counter() - started)

        if not results:
            metrics.empty_results.inc()
        metrics.requests.inc()
        return {
            "results": [{"item_id": item_id, "score": score} for item_id, score in results],
            "bundle_version": loaded_bundle.metadata.get("data_hash"),
        }

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def _load_bundle_from_path(bundle_path: Path | None) -> StudentBundle | None:
    configured_path = bundle_path or _path_from_env()
    if configured_path is None:
        return None
    return load_student_bundle(configured_path)


def _path_from_env() -> Path | None:
    value = os.getenv("LTRD_BUNDLE_PATH")
    return None if not value else Path(value)


def _query_example(query_features: dict[str, FeatureValue]) -> RankingExample:
    return RankingExample(
        query_id="serving-query",
        item_id="serving-dummy-item",
        group_id="serving-dummy-group",
        label=0.0,
        is_unbiased=False,
        position=None,
        features=query_features,
    )


class _ServingMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "ltrd_rank_requests_total",
            "Total rank requests.",
            registry=registry,
        )
        self.errors = Counter(
            "ltrd_rank_errors_total",
            "Total rank request errors.",
            ["kind"],
            registry=registry,
        )
        self.empty_results = Counter(
            "ltrd_rank_empty_results_total",
            "Total rank requests returning no results.",
            registry=registry,
        )
        self.latency = Histogram(
            "ltrd_rank_latency_seconds",
            "Rank request latency in seconds.",
            registry=registry,
        )
        self.item_count = Gauge(
            "ltrd_bundle_items",
            "Items loaded in the serving bundle.",
            registry=registry,
        )
