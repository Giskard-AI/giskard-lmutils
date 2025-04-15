default: help;

help: ## Display commands help
	@grep -E '^[a-zA-Z][a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY:

setup: ## Install dependencies
	rye sync
	if command -v black >/dev/null 2>&1; then echo "black already installed"; else rye install black; fi
	if command -v isort >/dev/null 2>&1; then echo "isort already installed"; else rye install isort; fi
	if command -v ruff >/dev/null 2>&1; then echo "ruff already installed"; else rye install ruff; fi
.PHONY: setup

format: ## Format code
	black src tests
	isort src tests
.PHONY: format

check_format: ## Check format
	black --check src tests
	isort --check src tests
.PHONY: check_format

check_linting: ## Check linting
	ruff check src tests
.PHONY: check_linting