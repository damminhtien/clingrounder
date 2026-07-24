.PHONY: install install-dev pre-commit lint type test test-fast test-full test-all test-targeted validate pipeline evaluate profile pipeline-report phase1-submit phase1-validate loop ablation qa qa-full clean

PYTHON ?= .venv/bin/python
RUN_ROOT ?= outputs/runs

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

test-fast: test

test-full:
	$(PYTHON) -m pytest -o addopts='' -m "not private and not model" tests

test-all:
	$(PYTHON) -m pytest -o addopts='' tests

test-targeted:
	$(PYTHON) -m pytest tests/test_schema.py tests/test_offset_mapping.py tests/test_kg_constraints.py -q

pipeline:
	$(PYTHON) -m medical_kg_nlp.cli pipeline run --input data/samples/sample_notes.jsonl --output outputs/predictions.jsonl

validate:
	$(PYTHON) -m medical_kg_nlp.cli validate --pred outputs/predictions.jsonl --documents data/samples/sample_notes.jsonl --dictionary data/dictionaries/seed_concepts.jsonl

evaluate:
	$(PYTHON) -m medical_kg_nlp.cli evaluate --gold data/samples/gold.jsonl --pred outputs/predictions.jsonl

profile:
	$(PYTHON) scripts/profile_data.py --documents data/samples/sample_notes.jsonl --gold data/samples/gold.jsonl --output outputs/profiles/sample_profile.json --markdown outputs/profiles/sample_profile.md

pipeline-report:
	$(PYTHON) scripts/evaluate_pipeline_steps.py --documents data/samples/sample_notes.jsonl --gold data/samples/gold.jsonl --dictionary data/dictionaries/seed_concepts.jsonl --output-dir outputs/evaluation/sample

phase1-submit:
	$(PYTHON) -m medical_kg_nlp.cli benchmark phase1 submission --input-dir data/raw/input --output-dir outputs/phase1/output --zip outputs/phase1/output.zip

phase1-validate:
	$(PYTHON) scripts/validate_phase1_submission.py --input-dir data/raw/input --output-dir outputs/phase1/output --zip outputs/phase1/output.zip --expected-count 100

loop: pipeline-report
	$(PYTHON) scripts/loop_engineer.py --current-report outputs/evaluation/sample/metrics.json --output-dir outputs/loops/sample --experiment-id BASELINE --module evaluation --hypothesis "Establish a valid end-to-end baseline." --change "Run current pipeline and generate stage-wise metrics."

ablation:
	$(PYTHON) scripts/run_ablation.py --config configs/ablations.yaml --run-root $(RUN_ROOT)

qa: lint type test

qa-full: lint type test-full

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist *.egg-info
