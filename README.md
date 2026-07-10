# learning-to-rank-distillation

Dataset-agnostic tooling for ranking-model distillation and marketplace-aware reranking.

v1.0 is scoped by [`REQUIREMENTS.md`](REQUIREMENTS.md). Future work lives in
[`ROADMAP.md`](ROADMAP.md).

## Motivation

This project started as a focused way to close two gaps for an Expedia Senior Director, ML/AI
interview loop: ranking-model distillation and multi-objective marketplace ranking. The code is
intentionally built as reusable infrastructure instead of an Expedia-only script. Amazon ESCI is the
primary public real-data flow because it has a query-candidate-relevance shape that maps cleanly to
learning-to-rank. Expedia RecTour remains the secondary travel-marketplace target, and all downstream
code consumes the shared `RankingExample` schema.

## Architecture

![Architecture diagram](docs/architecture_diagram.png)

The main flow is:

1. Dataset adapter maps raw rows into `RankingExample`.
2. Teacher trains a LightGBM LambdaMART ranker.
3. Student trains a PyTorch two-tower model, either with response-based KD or label-only no-KD.
4. Benchmark compares quality, latency, and size.
5. Fairness layer sweeps exposure floors and plots relevance vs. exposure fairness.
6. Promotion gate logs governed decisions to SQLite.

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
pytest tests/ -v
ruff check .
ruff format --check .
python -m learning_to_rank_distillation.benchmark.run_all
```

The CLI alias is also available after installation:

```bash
ltrd benchmark
ltrd benchmark --dataset esci --data-dir data/esci --limit 5000
ltrd train-teacher --dataset synthetic
ltrd train-teacher --dataset esci --data-dir data/esci --limit 5000
ltrd generate-synthetic-rectour --output-path data/synthetic/rectour_like.csv
```

## Benchmark Table

Example local run on the enriched synthetic fallback fixture:

| model | NDCG@5 | NDCG@10 | size bytes | p50 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| teacher-lightgbm | 0.3763 | 0.5237 | 111655 | 1.233 | 1.509 |
| student-no-kd-d16 | 0.4121 | 0.5515 | 35712 | 1.154 | 1.439 |
| student-kd-d8 | 0.4121 | 0.5515 | 33600 | 1.152 | 1.391 |
| student-kd-d16 | 0.4121 | 0.5515 | 35712 | 1.155 | 1.330 |
| student-kd-d32 | 0.4121 | 0.5515 | 39936 | 1.159 | 1.430 |

Generated artifacts:

- `artifacts/benchmark_table.json`
- `artifacts/quality_latency_pareto.png`
- `artifacts/fairness_tradeoff.json`
- `artifacts/fairness_tradeoff.png`
- `artifacts/promotion_registry.sqlite`

See [`docs/artifact_policy.md`](docs/artifact_policy.md) for what is committed versus ignored.

## Data

Amazon ESCI is the primary public-data path. Place the official shopping query files under
`data/esci/`:

- `shopping_queries_dataset_examples.csv` or `.parquet`
- `shopping_queries_dataset_products.csv` or `.parquet`
- `shopping_queries_dataset_sources.csv` or `.parquet` (optional)

The ESCI adapter maps `E/S/C/I` judgments to graded relevance `3/2/1/0`, merges product metadata
when available, derives conservative text-overlap features, and exposes the result through
`RankingExample`.

Real RecTour files should be placed under `data/rectour/`. The adapter validates actual files before
mapping rows into `RankingExample`. If the schema is missing or ambiguous, it raises a clear error
instead of guessing Expedia field names.

When real data is unavailable, the package uses deterministic synthetic ranking data so the models,
benchmark, fairness layer, and governance gate can still be developed and tested.

The synthetic fallback is RecTour-like rather than RecTour-derived. It is generated from public
schema/behavior descriptions: search-level features such as check-in/check-out dates, destination,
party size, point of sale, mobile flag, sort/filter settings; property-level features such as
`prop_id`, ratings, review count, price bucket, cancellation, ad and amenity flags; and behavioral
columns such as `num_clicks`, `is_trans`, `label`, `is_unbiased`, and observed `position`.
Use it to exercise the pipeline and inspect expected shapes, not to report Expedia benchmark claims.
It also borrows feature-family ideas from the 2013 ICDM Expedia hotel-search challenge: visitor
history, explicit price and historical price, property location scores, promotion flags, competitor
price/availability signals, and within-query rank features such as price/star/location rank.

## Design Decisions

- Ranking distillation: response-based KD is implemented first because it gives a clean, interpretable
  teacher-to-student control before adding feature- or relation-based losses.
- Latency-aware student: the student is a two-tower PyTorch model with precomputable item embeddings
  and a FAISS-backed item index wrapper.
- Marketplace ranking: exposure fairness is treated as a supply-side proxy, measured by top-k
  impression share for historically low-exposure groups plus exposure Gini.
- Governance discipline: promotion is executable policy, not prose. The default gate promotes only if
  NDCG@5 drop is at most 2% and p99 latency improves by at least 3x versus teacher.
- Dataset generality: ESCI and RecTour-specific mapping is isolated to `adapters/`; model,
  distillation, fairness, benchmark, and governance code use only `RankingExample`.

## What This Does Not Cover

- No online A/B test or live traffic evaluation.
- No actual partner revenue, commission, or negotiated exposure modeling. The fairness metric is only
  a proxy for supply-side exposure.
- No live serving endpoint in v1.0.
- No confirmed RecTour benchmark claims until real dataset access, version, schema, and any
  subsampling are documented here.
- Synthetic RecTour-like rows are hand-generated from public descriptions and do not reproduce
  Expedia's real traffic, supply, competition, pricing, or user-behavior distributions.
- Native FAISS search can conflict with Torch-linked OpenMP on this macOS environment, so the wrapper
  constructs the FAISS index but uses a stable NumPy inner-product search path in-process.
