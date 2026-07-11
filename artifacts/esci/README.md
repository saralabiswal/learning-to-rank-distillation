# Amazon ESCI Benchmark Artifacts

Author: Sarala Biswal

Generated from local Amazon ESCI files under `data/esci/`.

Run command:

```bash
python3 -m learning_to_rank_distillation.benchmark.run_all \
  --dataset esci \
  --data-dir data/esci \
  --limit 5000 \
  --student-epochs 2 \
  --output-dir artifacts/esci
```

The local run used the committed ESCI examples parquet and sources CSV plus the uncommitted local
products parquet. The products parquet is intentionally ignored because it is about 1 GB.

Reviewable outputs:

- `benchmark_table.json`
- `quality_latency_pareto.png`
- `fairness_tradeoff.json`
- `fairness_tradeoff.png`
- `fairness_pareto_frontier.json`
- `fairness_pareto_frontier.png`

Ignored local outputs:

- `models/`
- `promotion_registry.sqlite`
- `experiments.jsonl`
- `.matplotlib-cache/`
