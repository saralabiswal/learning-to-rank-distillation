# Learning-to-Rank Distillation for Marketplace Ranking

## Abstract

This project implements a production-shaped learning-to-rank system for search and marketplace
ranking. The core workflow is: load query-candidate relevance data, normalize it into a shared row
contract, train a strong teacher ranker, distill the teacher into smaller two-tower student models,
evaluate ranking quality against latency and size, inspect exposure fairness, and package a student
model for local serving.

The primary public dataset path is Amazon ESCI because it has the right shape for learning-to-rank:
queries, candidate products, and graded relevance judgments. Expedia RecTour is kept as a secondary
travel-marketplace adapter, guarded until real files are available. Synthetic marketplace data gives
the repo a deterministic smoke-test path, and MovieLens is included as an optional quickstart adapter
to show that the data contract is not tied to one domain.

The implementation is intentionally not a notebook-only experiment. It includes a CLI, Makefile,
CI smoke checks, benchmark artifacts, fairness plots, model bundle utilities, an index lifecycle, a
FastAPI serving endpoint, Prometheus metrics, Docker support, and a lightweight registry/promotion
path. Those pieces are small, but they force the project to address the same boundaries that appear
in real ranking systems: data contracts, model lineage, quality gates, serving shape, and monitoring.

## Business Problem

A marketplace ranking team has to solve more than "sort by relevance." The system must decide which
items are shown first for each user query while staying inside operational constraints. A strong
offline ranker can improve relevance, but it may be too costly or too awkward to serve directly.
It may depend on feature transformations, group-level list context, or model formats that are fine
offline but inconvenient for low-latency retrieval.

This creates a practical trade-off:

| Requirement | Ranking-system pressure |
|---|---|
| Relevance | Put the best candidates at the top for each query. |
| Latency | Return ranked results within a predictable p50/p99 budget. |
| Model size | Keep serving artifacts small enough to load and deploy repeatedly. |
| Supplier exposure | Track whether ranking changes concentrate impressions among a few groups. |
| Governance | Promote only when candidate quality, latency, and lineage are acceptable. |
| Portability | Keep dataset-specific assumptions out of the modeling and serving layers. |

The architecture addresses this by splitting the problem into a strong teacher path and a serving
student path. The teacher is optimized for quality. The student is optimized for retrieval and
serving. Distillation transfers behavior from the teacher into the student, and the benchmark
decides whether that transfer is useful enough to promote.

## System Architecture

The system is organized around a small number of contracts:

1. Dataset adapters produce `RankingExample` rows.
2. Feature vectorizers convert adapter-defined features into numeric tensors.
3. Teacher models score candidate items within query groups.
4. Student models learn from labels and optionally from teacher signals.
5. Benchmarks evaluate quality, latency, model size, fairness, and promotion checks.
6. Production utilities save a student bundle and serve it through an API.

The important boundary is that raw data schemas do not leak past `adapters/`. Once a row becomes a
`RankingExample`, the rest of the code can run the same training, evaluation, fairness, governance,
and serving logic across ESCI, RecTour, synthetic data, or MovieLens.

## Dataset Contract

Every downstream component consumes this row shape:

| Field | Purpose |
|---|---|
| `query_id` | Groups candidates belonging to the same ranking request. |
| `item_id` | Identifies the candidate item being scored. |
| `features` | Adapter-defined feature dictionary. Values are normalized to bool, int, float, string, or null. |
| `label` | Graded relevance target used for ranking loss and NDCG. |
| `group_id` | Supply-side grouping key used for exposure metrics. |
| `is_unbiased` | Marker for randomized or less-biased logging when available. |
| `position` | Optional 1-indexed logged position for inverse-propensity evaluation. |

The schema validates basic invariants: `query_id`, `item_id`, and `group_id` must be present,
`features` must be a dictionary, and `position` cannot be included as a training feature. That last
guard is important because observed rank position is an evaluation/logging field, not a feature the
model should learn from by default.

### Amazon ESCI Adapter

The ESCI adapter maps Amazon Shopping Queries data into the shared contract. It expects
`shopping_queries_dataset_examples` and can optionally merge `shopping_queries_dataset_products`
and `shopping_queries_dataset_sources`.

The adapter maps ESCI labels into graded relevance:

| ESCI label | Meaning | Relevance |
|---|---|---:|
| `E` | Exact | 3 |
| `S` | Substitute | 2 |
| `C` | Complement | 1 |
| `I` | Irrelevant | 0 |

When product metadata is available, the adapter merges title, description, bullet points, brand,
color, locale, and source metadata. It then derives conservative text features such as query token
count, title token count, description token count, and token-overlap ratios between the query and
product text fields. It also adds a stable product-id hash bucket so the student item tower still
has item-side signal when the large product metadata file is absent.

### RecTour, Synthetic, and MovieLens

The RecTour adapter is guarded. It validates the presence of real files and required mapping fields
instead of guessing private Expedia schema names. That keeps the project honest: no RecTour benchmark
claim is made until real files and schema details are available.

The synthetic generator creates RecTour-like marketplace rows for deterministic smoke tests and
fairness stress tests. It models search-level features, property-level features, behavioral labels,
position fields, and configurable supply concentration, cold-start rate, and exposure skew. It is
useful for development, but it is not a substitute for Expedia traffic.

MovieLens maps users to query groups, movies to items, ratings to graded labels, and optional
movie metadata to features. Its purpose is portability and quickstart usage, not primary marketplace
evidence.

## Feature Vectorization

The feature layer is deliberately simple and deterministic. `FeatureVectorizer` inspects all feature
names in the training examples and splits them into numeric and categorical columns. Numeric values
are copied into float arrays with missing values as zero. Categorical values are one-hot encoded
from observed training categories.

The two-tower student needs query-side and item-side features. `TwoTowerVectorizer` infers query-side
features by checking which features are constant within every query group. Features that vary within
a query are treated as item-side features. This is not a universal feature-store strategy, but it is
a pragmatic dataset-agnostic rule that lets the student train without hard-coded ESCI, RecTour, or
MovieLens feature names.

## Training Pipeline

The benchmark uses query-grouped splits. Rows from the same query stay together so the model is not
trained on some candidates from a query and tested on other candidates from that same query. This is
important for ranking because candidate lists carry query-level context.

The default benchmark flow is:

1. Load ranking examples from the selected dataset.
2. Split by query into train, validation, and test partitions.
3. Train a LightGBM LambdaMART teacher on the training query groups.
4. Score train and test rows with the teacher.
5. Train a label-only no-KD student as the control.
6. Train response-distilled students with several embedding dimensions.
7. Evaluate all models on held-out query groups.
8. Write benchmark rows, plots, fairness outputs, and promotion logs.

The teacher uses LightGBM's `LGBMRanker` with the `lambdarank` objective. Training examples are
ordered by query group and passed with group sizes, so LightGBM optimizes ranking within query lists.
The teacher saves both the model file and metadata containing model type, feature dimension, random
state, hyperparameters, and a stable hash of the training examples.

The student is a PyTorch two-tower ranker. One tower encodes query features, the other tower encodes
item features, and the score is the dot product of normalized embeddings. This shape is chosen
because item embeddings can be precomputed and indexed. That is the serving-oriented reason to
distill into a student rather than simply using the teacher everywhere.

## Distillation Methods

The repo compares distillation against a label-only baseline. This matters because a student that
does not beat its no-KD control is not demonstrating useful distillation, even if it trains without
errors.

### No-KD Baseline

The no-KD baseline trains the two-tower student directly against relevance labels using a listwise
label loss. It answers the control question: how much can the student learn without teacher help?

### Response-Based Distillation

Response-based KD matches the teacher's output distribution within a query list. The implemented loss
combines:

- KL divergence between softened student and teacher score distributions.
- A supervised listwise label loss against graded relevance labels.

The blend is controlled by `alpha`, and score smoothing is controlled by `temperature`. This is the
main distillation path used in the end-to-end benchmark because it works with the LightGBM teacher's
ordinary score outputs.

### Feature-Based Distillation

Feature-based KD trains the student item tower to match teacher-side item representations. The
implemented loss combines normalized embedding MSE with the supervised label loss. Teacher
representations are projected deterministically to the student embedding dimension, so benchmark
runs remain reproducible even when teacher and student representation sizes differ.

This path is used in the ablation runner with the transformer teacher because tree models do not
naturally expose the same kind of dense intermediate representation.

### Relation-Based Distillation

Relation-based KD preserves pairwise/listwise ordering structure. For each query group, the loss
compares teacher and student pairwise score differences using a tanh-transformed relation matrix.
This encourages the student to learn not just the teacher's absolute scores, but the relative
ordering relationships among candidates.

The ablation runner trains no-KD, response-KD, feature-KD, and relation-KD on the same split. That
turns distillation into an empirical comparison rather than a claim that every KD method is useful
in every setting.

## Evaluation

The primary quality metrics are mean NDCG@5 and NDCG@10 over query groups. For each query, candidates
are sorted by model score, discounted cumulative gain is computed from graded labels, and the result
is normalized against the ideal ordering for that query.

The benchmark also records:

| Metric | Implementation detail |
|---|---|
| p50 latency | Repeated in-process prediction timing over the test examples. |
| p99 latency | High-percentile timing from repeated prediction calls. |
| model size | Teacher artifact bytes or estimated student parameter bytes. |
| artifact hash | Stable content hash for training-example lineage. |
| promotion decision | Candidate-vs-teacher comparison logged by the promotion gate. |

The committed benchmark artifacts are intentionally lightweight examples, generated from the
deterministic synthetic path unless a run explicitly states otherwise. Local ESCI runs can be
produced with `make benchmark-esci` after placing the official data under `data/esci/`.

## Off-Policy Evaluation

Logged rankings are often biased by the previous ranking policy: items shown higher receive more
attention, clicks, and purchases. The `evaluation.ips` module provides inverse-propensity NDCG for
datasets that include observed positions.

The implementation estimates position propensities from observed position counts, clips them by a
minimum propensity, and weights relevance gains by the inverse propensity. This is not a replacement
for randomized experiments, but it gives the repo a path for separating ordinary relevance evaluation
from position-biased log evaluation when the dataset provides enough logging metadata.

## Marketplace Exposure Fairness

The fairness layer treats `group_id` as a supply-side exposure key. In ESCI this can be derived from
brand when available; in synthetic data it can represent marketplace suppliers or properties. The
system computes historical exposure shares from the training split, identifies low-exposure groups
from the bottom exposure quantile, and measures how much top-k impression share those groups receive
after reranking.

Two reranking strategies are implemented:

1. A constrained reranker that reserves enough slots for low-exposure groups to satisfy an exposure
   floor when such candidates are available.
2. A scalarized Pareto sweep that adds a continuous exposure bonus to standardized relevance scores.

The Pareto search reports NDCG@5, low-exposure impression share, exposure Gini@5, and whether each
operating point is non-dominated. This makes the trade-off visible. The goal is not to claim that
this proxy captures all marketplace fairness concerns. The goal is to make exposure a first-class
evaluation dimension instead of an afterthought.

## Governance and Promotion

Promotion is executable policy. The benchmark identifies a student candidate and compares it against
the teacher using quality, latency, and lineage checks. The default policy is intentionally strict:
a student should not lose more than the allowed NDCG threshold and should improve p99 latency enough
to justify replacing the teacher in a serving path.

Promotion decisions are logged locally, and CI runs a looser smoke version to verify that the gate
can execute in a clean environment. This is a small version of a larger production pattern: model
promotion should be a reproducible check, not an informal README claim.

## Production Shape

The production package turns a trained student into a local serving artifact. A student bundle
contains:

| File or metadata | Purpose |
|---|---|
| `student.pt` | PyTorch model weights. |
| `vectorizer.pkl` | Fitted query/item vectorizers. |
| `item_embeddings.npy` | Precomputed item embeddings for retrieval. |
| `items.json` | Item IDs, group IDs, and item features. |
| `metadata.json` | Model config, feature names, dimensions, metrics, training config, and data hash. |

The index lifecycle utility can build, validate, version, and publish a bundle by writing a `CURRENT`
pointer. The bundle can create a FAISS inner-product index when FAISS is available, and it falls back
to stable NumPy search when native FAISS search is not enabled or not suitable for the local runtime.

The FastAPI app exposes:

| Endpoint | Behavior |
|---|---|
| `GET /health` | Reports whether a bundle is loaded and how many items are available. |
| `POST /rank` | Encodes query features, searches precomputed item embeddings, and returns top-k item scores. |
| `GET /metrics` | Exposes Prometheus-format counters, gauges, and latency histograms. |

The Dockerfile and docker-compose service package the API, but no default bundle is committed. A
bundle must be built locally before `make serve`, `ltrd-serve`, or Docker Compose can return ranked
results.

## Current Results and Artifacts

The repository commits lightweight artifacts that make the project inspectable from a fresh clone:

- `artifacts/benchmark_table.json`
- `artifacts/distillation_ablation.json`
- `artifacts/quality_latency_pareto.png`
- `artifacts/fairness_tradeoff.png`
- `artifacts/fairness_pareto_frontier.png`
- `artifacts/cross_dataset/cross_dataset_benchmark.json`

Those artifacts should be read as smoke-test evidence unless the file or run notes explicitly state
that real data was used. The ESCI examples parquet and sources CSV are committed because they are
manageable. The ESCI products parquet is intentionally not committed because it is large and belongs
in a local data directory or Git LFS-backed storage.

## Limitations

This project is a production-shaped offline ranking system, not a deployed marketplace. The current
limits are:

- No online A/B testing or live traffic loop.
- No real partner economics, commissions, or negotiated exposure constraints.
- No confirmed RecTour benchmark claim until real RecTour files and schema details are available.
- No hosted model registry; the registry implementation is filesystem-based.
- No committed serving bundle; bundles are local generated artifacts.
- No distributed training or large-scale feature store integration.
- Synthetic data is useful for smoke tests and stress tests, but it does not reproduce real Expedia
  or Amazon production distributions.

The value of the repo is that these limitations are explicit while the interfaces are in place. The
same boundaries used here, dataset contract, distillation benchmark, fairness trade-off, promotion
gate, bundle format, index lifecycle, and serving endpoint, are the boundaries that would need to be
hardened for a full production deployment.
