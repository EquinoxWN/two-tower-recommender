.PHONY: setup lint test run ablation bench audit ci

PY ?= python

setup:
	$(PY) -m pip install --upgrade pip && $(PY) -m pip install -e ".[dev]"

# ruff, ruff format and mypy --strict.
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy

# 23 tests on synthetic data with planted preferences (no download needed).
test:
	$(PY) -m pytest -q

# Download MovieLens 1M (checksum-verified, into .tmp/data), train, index and evaluate.
run:
	twotower

# The same run without the logQ popularity correction.
ablation:
	twotower --no-logq

bench:
	@echo "M3: latency per stage behind FastAPI with ONNX Runtime, and MovieLens 25M"

# Known vulnerabilities in the installed Python dependencies.
audit:
	$(PY) -m pip_audit --skip-editable --cache-dir .tmp/pip-audit

ci: setup lint test
