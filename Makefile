PYTHON ?= python3
RUFF ?= ruff

.PHONY: install lint format test benchmark benchmark-esci clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUFF) check --no-cache src tests
	$(RUFF) format --check src tests

format:
	$(RUFF) format src tests

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

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf artifacts/.matplotlib-cache artifacts/models artifacts/*.sqlite
