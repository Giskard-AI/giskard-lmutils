default: help;

help: ## Display commands help
	@grep -E '^[a-zA-Z][a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY:

setup: ## Install dependencies
	uv sync --all-extras
.PHONY: setup

format: ## Format code
	uv run -- black src tests
	uv run -- isort src tests --profile black
.PHONY: format

check_format: ## Check format
	uv run -- black --check src tests
	uv run -- isort --check src tests --profile black
.PHONY: check_format

check_linting: ## Check linting
	uv run -- ruff check src tests
.PHONY: check_linting

test: ## Run tests
	uv run -- pytest tests
.PHONY: test