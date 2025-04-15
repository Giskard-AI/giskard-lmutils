default: help;

help: ## Display commands help
	@grep -E '^[a-zA-Z][a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY:

setup: ## Install dependencies
	rye sync --all-features
.PHONY: setup

format: ## Format code
	rye run black src tests
	rye run isort src tests
.PHONY: format

check_format: ## Check format
	rye run black --check src tests
	rye run isort --check src tests
.PHONY: check_format

check_linting: ## Check linting
	rye run ruff check src tests
.PHONY: check_linting

test: ## Run tests
	rye run pytest tests
.PHONY: test