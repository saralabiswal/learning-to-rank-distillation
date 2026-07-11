# learning-to-rank-distillation

[![CI](https://github.com/saralabiswal/learning-to-rank-distillation/actions/workflows/ci.yml/badge.svg)](https://github.com/saralabiswal/learning-to-rank-distillation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Status](https://img.shields.io/badge/status-production--shaped-1a3a2a)

**A governed ranking lifecycle for search and marketplace ranking — from teacher training through promotion-gated serving.**

Not a model notebook. A full lifecycle: train a strong offline teacher, distill it into a
latency-safe student, prove the trade-off with fairness and promotion evidence, and package the
result into a servable, monitored bundle.

For a visual walkthrough, start with [`docs/learning_flow.html`](docs/learning_flow.html) or
[`docs/learning_guide.md`](docs/learning_guide.md). For implementation depth, see
[`docs/technical_writeup.md`](docs/technical_writeup.md).

---

## The Problem

Ranking teams rarely fail on model quality. They fail on the decisions around the model — the ones
that never show up in an offline NDCG number.

| Failure mode | What actually happens |
|---|---|
| **The serving trap** | The best offline ranker is too large or too slow to serve at p99 latency budgets. Teams either ship a model they can't afford to run, or hand-roll a smaller one with no formal link back to the teacher's ranking behavior. |
| **The silent exposure shift** | A ranking change that improves relevance can quietly starve specific suppliers or sellers of exposure. Without a fairness measurement built into the evaluation loop, that shift isn't caught until a business stakeholder notices. |
| **The promotion-by-vibes problem** | "Does this look better?" is not a promotion policy. Without an executable gate, model promotion becomes a judgment call re-litigated on every release. |

This repo treats ranking as a governed lifecycle, not a single training script. Every dataset, model,
and evaluation stage feeds the same production question: **is this model good enough, fast enough,
and fair enough to serve — and can that decision be defended?**

## How The Architecture Solves It

Domain-specific data mapping is isolated from model training, evaluation, and serving. Each raw
dataset is converted into a shared `RankingExample` schema, so the same teacher, student, fairness,
governance, and production code runs unmodified against ESCI, RecTour, synthetic, or MovieLens data.

| Stage | Component | Purpose |
|---|---|---|
| 1 | Dataset adapter | Maps raw rows into `RankingExample` — domain logic stays out of the model layer |
| 2 | Teacher | LightGBM LambdaMART ranker optimized purely for ranking quality |
| 3 | Student | PyTorch two-tower model with precomputable item embeddings, built for serving |
| 4 | Distillation | Transfers teacher ranking behavior into the smaller student (response, feature, or relation-based KD) |
| 5 | Benchmark | Quantifies the quality/latency/size trade-off — NDCG@5/10, model size, p50/p99 latency |
| 6 | Fairness layer | Sweeps exposure floors, plots relevance vs. exposure fairness for low-exposure supply groups |
| 7 | Promotion gate | Logs a governed, executable promote/reject decision to SQLite |

## Architecture

![Architecture diagram](docs/architecture_diagram.png)

For the full pictorial walkthrough — data adapters, teacher-student distillation, evaluation
metrics, fairness trade-offs, and production lifecycle — see
[`docs/learning_flow.html`](docs/learning_flow.html).

## Who This Is For

| Audience | Why it matters |
|---|---|
| **Search / marketplace ranking teams** | A reference lifecycle for taking a ranking model from offline training to a governed, servable decision |
| **ML platform engineers** | A worked example of promotion-as-policy: executable gates instead of ad hoc release judgment |
| **Applied scientists evaluating KD strategies** | A head-to-head ablation of response-, feature-, and relation-based distillation on the same harness |

---

## Quickstart

```bash
make install
# or directly:
pip install -e ".[dev]"

# Optional dashboard:
pip install -e ".[dev,dashboard]"
```

```bash
make lint
make test
make benchmark
make ablation
make cross-dataset
make promotion-check
```

```bash
ltrd benchmark
ltrd benchmark --dataset esci --data-dir data/esci --limit 5000
ltrd distillation-ablation
ltrd train-teacher --dataset synthetic
ltrd train-teacher --dataset esci --data-dir data/esci --limit 5000
ltrd generate-synthetic-rectour --output-path data/synthetic/rectour_like.csv
make dashboard
make benchmark-esci
```

Serving requires a previously built student bundle at `artifacts/bundles/current`. No bundle is
committed to the repo.

```bash
make serve
ltrd-serve --bundle-path artifacts/bundles/current
docker compose up --build
```

---

## Benchmark Results

Local run on the enriched synthetic fallback fixture:

| Model | NDCG@5 | NDCG@10 | Size (bytes) | p50 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|---:|
| teacher-lightgbm | 0.3763 | 0.5237 | 111,705 | 1.321 | 1.642 |
| student-no-kd-d16 | 0.4181 | 0.5543 | 36,096 | 1.178 | 1.480 |
| student-kd-d8 | 0.4121 | 0.5482 | 33,984 | 1.174 | 1.461 |
| student-kd-d16 | 0.4121 | 0.5482 | 36,096 | 1.182 | 1.376 |
| student-kd-d32 | 0.4121 | 0.5515 | 40,320 | 1.219 | 1.387 |

## Distillation Ablation

Local run on the synthetic fallback fixture with the transformer teacher:

| Model | NDCG@5 | NDCG@10 | Final loss |
|---|---:|---:|---:|
| teacher-transformer | 0.4004 | 0.5234 | — |
| student-no-kd-d16 | 0.2685 | 0.4628 | 1.7857 |
| student-response-kd-d16 | 0.2921 | 0.4863 | 0.5361 |
| student-feature-kd-d16 | 0.3534 | 0.4764 | 0.9500 |
| student-relation-kd-d16 | 0.4038 | 0.5981 | 0.7192 |

**Reading the result:** on this fixture, relation-based KD wins on NDCG because preserving the
teacher's pairwise/listwise ordering matters more than matching softened scores alone. Feature-based
KD also beats the no-KD control; response-based KD mainly reduces training loss without a matching
NDCG gain. These are smoke-test results on synthetic data, not claims about real-data performance.

Tracked example artifacts: `artifacts/benchmark_table.json`,
`artifacts/cross_dataset/cross_dataset_benchmark.json`, `artifacts/distillation_ablation.json`,
`artifacts/quality_latency_pareto.png`, `artifacts/fairness_tradeoff.json`,
`artifacts/fairness_tradeoff.png`, `artifacts/fairness_pareto_frontier.json`,
`artifacts/fairness_pareto_frontier.png`.

Real Amazon ESCI benchmark artifacts can be generated under `artifacts/esci/` with
`make benchmark-esci`. The committed top-level artifacts remain the deterministic synthetic smoke
examples unless explicitly regenerated. See [`docs/artifact_policy.md`](docs/artifact_policy.md) for
the full committed-vs-ignored policy.

---

## Data

Amazon ESCI is the primary public-data path. Place the official shopping query files under
`data/esci/`:

- `shopping_queries_dataset_examples.csv` or `.parquet`
- `shopping_queries_dataset_products.csv` or `.parquet`
- `shopping_queries_dataset_sources.csv` or `.parquet` (optional)

This repo commits the official examples parquet and sources CSV downloaded from Amazon Science on
July 10, 2026. The products parquet (~1.03 GB, requires Git LFS) is not committed; the ESCI adapter
still runs without it using query fields, sources, locale, and a stable product-id hash bucket, and
merges product metadata automatically when the file is present. The adapter maps `E/S/C/I` judgments
to graded relevance `3/2/1/0` and derives conservative text-overlap features.

Real RecTour files go under `data/rectour/`. The adapter validates real files before mapping rows and
raises a clear error on missing or ambiguous schema instead of guessing Expedia field names. The
synthetic fallback is **RecTour-like, not RecTour-derived** — generated from public schema/behavior
descriptions and borrowed feature-family ideas from the 2013 ICDM Expedia hotel-search challenge. Use
it to exercise the pipeline and inspect expected shapes, not to report Expedia benchmark claims.
`SyntheticMarketplaceConfig` can vary supply concentration, cold-start rate, and logged exposure skew
for stress testing.

MovieLens is available as a third adapter under `data/movielens/` (`ratings.csv`, optional
`movies.csv`), mapping users to ranking queries, movies to items, ratings to graded labels, and
genre/year to features.

Use `synthetic` when real data is unavailable. Real-data adapters fail fast on missing or ambiguous
files; the synthetic path keeps models, benchmark, fairness, and governance testable without external
data access.

## Production Shape

The `production/` package provides the deployable skeleton:

| Capability | Implementation |
|---|---|
| Bundle packaging | Model weights, vectorizer, item embeddings, metrics, config, and data hash |
| Bundle lifecycle | Versioned bundle/index lifecycle with validation and a `CURRENT` publish pointer |
| Serving API | FastAPI endpoint (`/health`, `/rank`, `/metrics`) backed by precomputed item embeddings |
| Metrics | Prometheus counters for request count, errors, empty results, latency, loaded item count |
| Experiment tracking | JSONL tracking with optional MLflow logging (`LTRD_TRACKING_BACKEND=mlflow`) |
| Registry | Filesystem model registry for bundle versions and promotion stages |
| Deployment | Dockerfile, docker-compose service, `loadtest/k6-ranking.js` |

No default serving bundle is committed. Build one with the `production.bundle` or
`production.index_lifecycle` APIs before running `make serve`, `ltrd-serve`, or Docker Compose.

## Design Decisions

| Decision | Rationale |
|---|---|
| Response-based KD as the primary control | Simple, stable baseline; feature- and relation-based KD available via the transformer-teacher ablation path |
| Two-tower student architecture | Precomputable item embeddings plus a FAISS-backed item index wrapper keep serving latency low |
| Exposure fairness as a supply-side proxy | Measured via top-k impression share for historically low-exposure groups plus exposure Gini |
| Multi-objective search | Benchmark writes both a constrained exposure-floor sweep and a scalarized relevance/fairness Pareto search, marking non-dominated operating points |
| IPS evaluation | `evaluation.ips` provides clipped inverse-propensity NDCG for logged rows with observed positions, isolating position bias from standard NDCG |
| Governance as executable policy | Default gate promotes only if NDCG@5 drop ≤ 2% and p99 latency improves ≥ 3x versus teacher |
| CI enforcement | GitHub Actions runs lint, tests, a benchmark smoke run, and a promotion-gate smoke check; local defaults stay strict, CI uses looser smoke thresholds to absorb runner latency noise |
| Dataset generality | ESCI/RecTour-specific mapping is isolated to `adapters/`; model, distillation, fairness, benchmark, and governance code depend only on `RankingExample` |

## What This Does Not Cover

- No online A/B test or live traffic evaluation.
- No actual partner revenue, commission, or negotiated exposure modeling — the fairness metric is a
  supply-side proxy only.
- No hosted production deployment or online traffic integration. The FastAPI endpoint is a local
  serving skeleton around saved bundles.
- No confirmed RecTour benchmark claims until real dataset access, version, schema, and any
  subsampling are documented here.
- Synthetic RecTour-like rows do not reproduce Expedia's real traffic, supply, competition, pricing,
  or user-behavior distributions.
- FAISS search is opt-in (`LTRD_USE_FAISS_SEARCH=1`); the default in-process path uses stable NumPy
  inner-product search because FAISS/Torch OpenMP conflicts can abort this macOS environment.
