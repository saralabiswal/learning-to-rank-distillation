PYTHON ?= python3
RUFF ?= ruff

.PHONY: install lint format test benchmark benchmark-esci ablation cross-dataset dashboard serve load-test promotion-check clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUFF) check --no-cache src tests apps
	$(RUFF) format --check src tests apps

format:
	$(RUFF) format src tests apps

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -p no:cacheprovider tests -v

benchmark:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m learning_to_rank_distillation.benchmark.run_all \
		--output-dir artifacts

benchmark-esci:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m learning_to_rank_distillation.benchmark.run_all \
		--dataset esci \
		--data-dir data/esci \
		--limit 5000 \
		--output-dir artifacts

ablation:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m learning_to_rank_distillation.benchmark.distillation_ablation \
		--output-dir artifacts

cross-dataset:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m learning_to_rank_distillation.benchmark.cross_dataset \
		--output-dir artifacts/cross_dataset

dashboard:
	streamlit run apps/pareto_dashboard.py

serve:
	ltrd-serve --bundle-path artifacts/bundles/current --host 127.0.0.1 --port 8000

load-test:
	k6 run loadtest/k6-ranking.js

promotion-check:
	$(PYTHON) -m learning_to_rank_distillation.benchmark.promotion_check \
		--benchmark-table artifacts/benchmark_table.json \
		--registry-path artifacts/promotion_ci.sqlite \
		--max-ndcg-drop 0.50 \
		--min-latency-improvement 0.25

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf artifacts/.matplotlib-cache artifacts/models artifacts/*.sqlite
