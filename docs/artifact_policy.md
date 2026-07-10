# Artifact Policy

This repo keeps lightweight, reviewable artifacts that make the project legible from a clone. It
does not commit raw datasets, trained model binaries, local registries, or cache directories.

## Committed

- `docs/architecture_diagram.png`
- `artifacts/benchmark_table.json`
- `artifacts/cross_dataset/cross_dataset_benchmark.json`
- `artifacts/distillation_ablation.json`
- `artifacts/quality_latency_pareto.png`
- `artifacts/fairness_tradeoff.json`
- `artifacts/fairness_tradeoff.png`
- `artifacts/fairness_pareto_frontier.json`
- `artifacts/fairness_pareto_frontier.png`
- `data/esci/.gitkeep`
- `data/esci/SOURCE.md`
- `data/esci/shopping_queries_dataset_examples.parquet`
- `data/esci/shopping_queries_dataset_sources.csv`
- `data/rectour/.gitkeep`

The benchmark artifacts are generated from the deterministic synthetic fallback unless a run clearly
states that it used real data.

## Ignored

- Raw data under `data/esci/` and `data/rectour/`, except the committed ESCI examples parquet and
  sources CSV documented in `data/esci/SOURCE.md`
- Generated synthetic CSVs under `data/synthetic/`
- Student serving bundles under `artifacts/bundles/`
- Trained model files under `artifacts/models/`
- Nested per-dataset benchmark artifacts under `artifacts/cross_dataset/*/`
- Local promotion registries such as `artifacts/*.sqlite`
- Local model registries such as `artifacts/model_registry.json`
- Experiment logs such as `artifacts/**/experiments.jsonl`
- Local caches such as `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, and
  `artifacts/.matplotlib-cache/`

## Regeneration

Use the synthetic smoke path for committed benchmark artifacts:

```bash
make benchmark
make ablation
```

Use the real ESCI path only after placing the official files under `data/esci/`:

```bash
make benchmark-esci
```
