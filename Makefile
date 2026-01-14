.PHONY: help install install-dev install-all clean lint format type-check test test-unit test-integration test-e2e coverage serve docs build publish

PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
BLACK := $(PYTHON) -m black
MYPY := $(PYTHON) -m mypy

# Default target
help:
	@echo "KV-Bench Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  install          Install package in production mode"
	@echo "  install-dev      Install package with development dependencies"
	@echo "  install-all      Install package with all optional dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  lint             Run ruff linter"
	@echo "  format           Format code with black and ruff"
	@echo "  type-check       Run mypy type checker"
	@echo "  check            Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-e2e         Run end-to-end tests only"
	@echo "  coverage         Run tests with coverage report"
	@echo ""
	@echo "Server:"
	@echo "  serve            Start the KV-Bench server"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Build:"
	@echo "  build            Build distribution packages"
	@echo "  clean            Clean build artifacts"

# Installation targets
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

install-all:
	$(PIP) install -e ".[all]"
	pre-commit install

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf coverage_html/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Code quality targets
lint:
	$(RUFF) check src/ tests/

format:
	$(BLACK) src/ tests/
	$(RUFF) check --fix src/ tests/

type-check:
	$(MYPY) src/kvbench/

check: lint type-check
	@echo "All checks passed!"

# Testing targets
test:
	$(PYTEST) tests/

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v -m integration

test-e2e:
	$(PYTEST) tests/e2e/ -v -m e2e

coverage:
	$(PYTEST) tests/ \
		--cov=src/kvbench \
		--cov-report=term-missing \
		--cov-report=html:coverage_html \
		--cov-report=xml:coverage.xml \
		--cov-branch
	@echo ""
	@echo "HTML report generated at coverage_html/index.html"

# Server target
serve:
	kvbench serve

# Documentation targets
docs:
	mkdocs build -f docs/mkdocs.yml

docs-serve:
	mkdocs serve -f docs/mkdocs.yml

# Build target
build: clean
	$(PYTHON) -m build

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files
