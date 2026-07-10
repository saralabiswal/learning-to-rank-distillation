# Artifact Policy

This repo keeps lightweight, reviewable artifacts that make the project legible from a clone. It
does not commit raw datasets, trained model binaries, local registries, or cache directories.

## Committed

- `docs/architecture_diagram.png`
- `artifacts/benchmark_table.json`
- `artifacts/quality_latency_pareto.png`
- `artifacts/fairness_tradeoff.json`
- `artifacts/fairness_tradeoff.png`
- `data/esci/.gitkeep`
- `data/rectour/.gitkeep`

The benchmark artifacts are generated from the deterministic synthetic fallback unless a run clearly
states that it used real data.

## Ignored

- Raw data under `data/esci/` and `data/rectour/`
- Generated synthetic CSVs under `data/synthetic/`
- Trained model files under `artifacts/models/`
- Local promotion registries such as `artifacts/*.sqlite`
- Local caches such as `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, and
  `artifacts/.matplotlib-cache/`

## Regeneration

Use the synthetic smoke path for committed benchmark artifacts:

```bash
make benchmark
```

Use the real ESCI path only after placing the official files under `data/esci/`:

```bash
make benchmark-esci
```
