"""Small transformer-based ranking teacher."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from learning_to_rank_distillation.benchmark.metrics import ndcg_at_k
from learning_to_rank_distillation.distillation.common import listwise_label_loss
from learning_to_rank_distillation.features import FeatureVectorizer
from learning_to_rank_distillation.models.teacher import content_hash
from learning_to_rank_distillation.schema import RankingExample, validate_examples


@dataclass(slots=True)
class TransformerRankerTeacher:
    """Listwise transformer teacher with candidate representation access.

    The model is intentionally small so it can run as a local research baseline alongside the
    LightGBM LambdaMART teacher.
    """

    random_state: int = 13
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 0.01
    epochs: int = 8
    auxiliary_weight: float = 0.15
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    model: _TransformerRanker | None = None

    def fit(self, examples: list[RankingExample]) -> TransformerRankerTeacher:
        """Fit the teacher with a listwise loss plus an auxiliary relevance regression head."""

        validate_examples(examples)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        torch.manual_seed(self.random_state)
        self.vectorizer.fit(examples)
        self.model = _TransformerRanker(
            input_dim=self.vectorizer.output_dim(),
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            self.model.train()
            for _, query_examples in _indexed_query_groups(examples).values():
                features = torch.from_numpy(self.vectorizer.transform(query_examples)).unsqueeze(0)
                labels = torch.from_numpy(
                    np.asarray([example.label for example in query_examples], dtype=np.float32)
                )
                optimizer.zero_grad()
                scores, auxiliary, _ = self.model(features)
                scores = scores.squeeze(0)
                auxiliary = auxiliary.squeeze(0)
                loss = listwise_label_loss(scores, labels) + self.auxiliary_weight * F.mse_loss(
                    auxiliary,
                    _standardize(labels),
                )
                loss.backward()
                optimizer.step()
        return self

    def predict(self, examples: list[RankingExample]) -> np.ndarray:
        """Score examples in the caller's input order."""

        if self.model is None:
            raise RuntimeError("teacher must be fitted before predict")
        self.model.eval()
        scores = np.zeros(len(examples), dtype=np.float32)
        with torch.no_grad():
            for indices, query_examples in _indexed_query_groups(examples).values():
                features = torch.from_numpy(self.vectorizer.transform(query_examples)).unsqueeze(0)
                query_scores, _, _ = self.model(features)
                scores[indices] = query_scores.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return scores

    def item_representations(self, examples: list[RankingExample]) -> np.ndarray:
        """Return transformer candidate representations in the caller's input order."""

        if self.model is None:
            raise RuntimeError("teacher must be fitted before representations")
        self.model.eval()
        representations = np.zeros((len(examples), self.hidden_dim), dtype=np.float32)
        with torch.no_grad():
            for indices, query_examples in _indexed_query_groups(examples).values():
                features = torch.from_numpy(self.vectorizer.transform(query_examples)).unsqueeze(0)
                _, _, hidden = self.model(features)
                representations[indices] = (
                    hidden.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
        return representations

    def evaluate(self, examples: list[RankingExample]) -> dict[str, float]:
        scores = self.predict(examples)
        return {
            "ndcg@5": ndcg_at_k(examples, scores, k=5),
            "ndcg@10": ndcg_at_k(examples, scores, k=10),
        }

    def save(self, output_dir: Path, training_examples: list[RankingExample]) -> Path:
        if self.model is None:
            raise RuntimeError("teacher must be fitted before save")
        output_dir.mkdir(parents=True, exist_ok=True)
        data_hash = content_hash(training_examples)
        model_path = output_dir / f"transformer-teacher-{data_hash[:12]}.pt"
        metadata_path = output_dir / f"transformer-teacher-{data_hash[:12]}.json"
        torch.save(self.model.state_dict(), model_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "model_type": "TransformerRankerTeacher",
                    "data_hash": data_hash,
                    "feature_dim": self.vectorizer.output_dim(),
                    "hidden_dim": self.hidden_dim,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "random_state": self.random_state,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return metadata_path


class _TransformerRanker(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.auxiliary_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(self.input_projection(features))
        scores = self.score_head(hidden).squeeze(-1)
        auxiliary = self.auxiliary_head(hidden).squeeze(-1)
        return scores, auxiliary, hidden


def _indexed_query_groups(
    examples: list[RankingExample],
) -> dict[str, tuple[list[int], list[RankingExample]]]:
    groups: dict[str, tuple[list[int], list[RankingExample]]] = {}
    for index, example in enumerate(examples):
        indices, query_examples = groups.setdefault(example.query_id, ([], []))
        indices.append(index)
        query_examples.append(example)
    return groups


def _standardize(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if float(std) == 0.0:
        return torch.zeros_like(values)
    return (values - values.mean()) / std
