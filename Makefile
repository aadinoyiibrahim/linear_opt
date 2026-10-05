# Common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
IMAGE ?= linear-opt

.PHONY: help install lint test test-large cov docs docs-serve app docker docker-app clean-cache

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install everything (all extras, dev and docs groups) and the git hooks
	uv sync --all-extras --group docs
	uv run pre-commit install

lint: ## Ruff (lint + format check) and mypy --strict
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test: ## Test suite (instances within the restricted Gurobi licence)
	uv run pytest

test-large: ## Large instances (needs a full Gurobi licence)
	uv run pytest -m large -v

cov: ## Tests with coverage (CI gate: 90%)
	uv run pytest --cov --cov-report=term --cov-fail-under=90

docs: ## Build the documentation site (strict)
	uv run mkdocs build --strict

docs-serve: ## Serve the docs at http://127.0.0.1:8000
	uv run mkdocs serve

app: ## Start the dashboard at http://localhost:8501
	uv run linopt app

docker: ## Build the Docker image
	docker build -t $(IMAGE) .

docker-app: docker ## Run the dashboard in Docker at http://localhost:8501
	docker run --rm -p 8501:8501 $(IMAGE) app --headless --address 0.0.0.0

clean-cache: ## Remove tool caches and __pycache__ folders
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis .coverage* coverage.xml site
