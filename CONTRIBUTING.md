# Contributing

Author: Sarala Biswal

## Local Setup

```bash
pip install -e ".[dev]"
make lint
make test
```

Use `make benchmark` to refresh the synthetic benchmark artifacts. Real dataset artifacts should
only be published when the data source, version, download date, and subsampling are documented.

## Code Expectations

- Keep dataset-specific mapping inside `src/learning_to_rank_distillation/adapters/`.
- Preserve query-grouped splits for training and evaluation.
- Add focused tests for new ranking, fairness, governance, or serving behavior.
- Do not commit raw datasets, trained model binaries, SQLite registries, cache directories, or local
  experiment logs.

## Pull Request Checklist

- `make lint`
- `make test`
- `make benchmark` when benchmark behavior or artifacts change
- Update `README.md` or the relevant public docs when scope or public claims change
