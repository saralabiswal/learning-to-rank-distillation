# Learning-to-Rank Distillation for Marketplace Ranking

## Abstract

This project implements a dataset-agnostic learning-to-rank distillation toolkit for marketplace
ranking. It starts with a LightGBM LambdaMART teacher and PyTorch two-tower student, then extends the
research surface with a transformer teacher, response-based distillation, feature-based
distillation, relation-based distillation, exposure-aware reranking, inverse-propensity evaluation,
and production-shaped serving artifacts.

## Dataset Contract

All adapters map raw rows into `RankingExample`: one query, one candidate item, adapter-defined
features, a graded relevance label, a supply-side `group_id`, and optional logged position metadata.
This keeps model, benchmark, fairness, governance, and serving code independent of raw dataset
schemas.

Amazon ESCI is the primary public real-data flow because query/product/judgment rows map directly to
learning-to-rank. Expedia RecTour remains the travel-marketplace target adapter and stays guarded
until real files are available. MovieLens is included as a third adapter to prove the contract is not
commerce-specific. The synthetic generator covers local smoke tests and configurable marketplace
stress tests such as concentrated supply, cold-start items, and exposure skew.

## Distillation Methods

The v1 baseline uses response-based KD: the student matches the teacher's softened output
distribution while retaining a supervised listwise loss against labels. Tier 2 adds two more
methods. Feature-based KD matches the student's item-tower embedding against a transformer teacher
candidate representation. Relation-based KD preserves pairwise score differences within each query.

The ablation runner trains all three methods on the same split, with the no-KD student as the
control. This is more informative than reporting a single best number because it shows whether
distillation improves ranking quality, merely lowers training loss, or fails on small/noisy query
lists.

## Marketplace Fairness

The fairness layer treats exposure as a supply-side proxy. It computes historical exposure by
`group_id`, identifies low-exposure groups, and evaluates top-k impression share plus exposure Gini.
Two reranking paths are implemented: a constrained exposure-floor sweep and a scalarized Pareto
search over relevance and exposure fairness. The Pareto output marks non-dominated operating points
so the trade-off can be inspected rather than hidden behind one chosen threshold.

## Evaluation and Governance

Standard held-out metrics use query-grouped splits and NDCG@5/NDCG@10. For logged data with observed
positions, `evaluation.ips` provides clipped inverse-propensity NDCG so biased logs can be evaluated
separately from ordinary relevance labels. Promotion is executable policy: a candidate is compared
with the teacher on quality, latency, lineage, and metric validity, then logged to SQLite. CI runs a
benchmark smoke check and a promotion-gate smoke check to make the governance path visible.

## Production Shape

The production layer saves a coherent student bundle containing model weights, vectorizer, item
embeddings, item metadata, metrics, config, and data hash. A lifecycle utility validates bundles,
versions them, and publishes a `CURRENT` pointer. The FastAPI service loads a bundle, exposes
`/rank`, `/health`, and `/metrics`, and serves retrieval from precomputed item embeddings.
Prometheus metrics cover request count, latency, errors, empty results, and item count. Docker,
docker-compose, and k6 load-test files provide a deployment and latency-test skeleton.

## Limitations

Current public benchmark artifacts are synthetic smoke results, not claims about Expedia traffic or
Amazon production search. Real ESCI and RecTour benchmark claims remain pending until the raw files,
download date, schema version, and subsampling are documented. The model registry is filesystem-based
rather than a hosted MLflow registry. The load-test script is present, but real p50/p95/p99 serving
numbers require building a bundle, running the API, and executing k6 on the target machine.
