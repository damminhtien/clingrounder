.PHONY: install install-dev pre-commit lint type test test-targeted validate pipeline evaluate profile ablation qa clean

PYTHON ?= .venv/bin/python

install:
	python -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

pre-commit:
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check .

type:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest tests

test-targeted:
	$(PYTHON) -m pytest tests/test_schema.py tests/test_offset_mapping.py tests/test_kg_constraints.py -q

pipeline:
	$(PYTHON) scripts/run_pipeline.py --input data/samples/sample_notes.jsonl --output outputs/predictions.jsonl

validate:
	$(PYTHON) scripts/validate_predictions.py --pred outputs/predictions.jsonl --documents data/samples/sample_notes.jsonl --dictionary data/dictionaries/seed_concepts.jsonl

evaluate:
	$(PYTHON) scripts/evaluate.py --gold data/samples/gold.jsonl --pred outputs/predictions.jsonl

profile:
	$(PYTHON) scripts/profile_data.py --documents data/samples/sample_notes.jsonl --gold data/samples/gold.jsonl --output outputs/profiles/sample_profile.json --markdown outputs/profiles/sample_profile.md

ablation:
	$(PYTHON) scripts/run_ablation.py --config configs/ablations.yaml

qa: lint type test

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist *.egg-info
