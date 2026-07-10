"""PyTorch two-tower student ranking model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from learning_to_rank_distillation.features import TwoTowerVectorizer
from learning_to_rank_distillation.schema import RankingExample


@dataclass(frozen=True, slots=True)
class StudentConfig:
    embedding_dim: int = 16
    hidden_dim: int = 32
    normalize_embeddings: bool = True
    random_state: int = 13


class TwoTowerRanker(nn.Module):
    """Query tower + item tower combined with dot product/cosine similarity."""

    def __init__(
        self,
        *,
        query_dim: int,
        item_dim: int,
        config: StudentConfig,
    ) -> None:
        super().__init__()
        self.normalize_embeddings = config.normalize_embeddings
        self.query_tower = _tower(query_dim, config.hidden_dim, config.embedding_dim)
        self.item_tower = _tower(item_dim, config.hidden_dim, config.embedding_dim)

    def encode_query(self, query_features: torch.Tensor) -> torch.Tensor:
        embeddings = self.query_tower(query_features)
        return _normalize(embeddings) if self.normalize_embeddings else embeddings

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        embeddings = self.item_tower(item_features)
        return _normalize(embeddings) if self.normalize_embeddings else embeddings

    def forward(self, query_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        query_embeddings = self.encode_query(query_features)
        item_embeddings = self.encode_item(item_features)
        return torch.sum(query_embeddings * item_embeddings, dim=1)


@dataclass(slots=True)
class TwoTowerStudent:
    """Fitted student wrapper with vectorization and prediction helpers.

    Implements FR-2.1.
    """

    config: StudentConfig = field(default_factory=StudentConfig)
    vectorizer: TwoTowerVectorizer = field(default_factory=TwoTowerVectorizer)
    model: TwoTowerRanker | None = None

    def initialize(self, examples: list[RankingExample]) -> TwoTowerStudent:
        torch.manual_seed(self.config.random_state)
        self.vectorizer.fit(examples)
        self.model = TwoTowerRanker(
            query_dim=self.vectorizer.query_dim(),
            item_dim=self.vectorizer.item_dim(),
            config=self.config,
        )
        return self

    def predict(self, examples: list[RankingExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("student must be initialized before predict")
        self.model.eval()
        with torch.no_grad():
            query_features, item_features = self.tensors(examples)
            scores = self.model(query_features, item_features)
        return scores.detach().cpu().numpy().astype(np.float32)

    def query_embeddings(self, examples: list[RankingExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("student must be initialized before query embeddings")
        self.model.eval()
        with torch.no_grad():
            query_features = torch.from_numpy(self.vectorizer.transform_query(examples))
            embeddings = self.model.encode_query(query_features)
        return embeddings.detach().cpu().numpy().astype(np.float32)

    def item_embeddings(self, examples: list[RankingExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("student must be initialized before item embeddings")
        self.model.eval()
        with torch.no_grad():
            item_features = torch.from_numpy(self.vectorizer.transform_item(examples))
            embeddings = self.model.encode_item(item_features)
        return embeddings.detach().cpu().numpy().astype(np.float32)

    def tensors(self, examples: list[RankingExample]) -> tuple[torch.Tensor, torch.Tensor]:
        query_features = torch.from_numpy(self.vectorizer.transform_query(examples))
        item_features = torch.from_numpy(self.vectorizer.transform_item(examples))
        return query_features, item_features

    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())

    def estimated_size_bytes(self) -> int:
        return self.parameter_count() * 4


def _tower(input_dim: int, hidden_dim: int, embedding_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, embedding_dim),
    )


def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)
