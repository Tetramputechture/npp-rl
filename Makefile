# Makefile for linting and fixing Python code in this repository
# - Lints recursively
# - Removes unused imports automatically when using the "fix" targets

SHELL := /bin/bash
PY_DIRS := npp_rl

.PHONY: help dev-setup lint fix imports

help:
	@echo "Available targets:"
	@echo "  make dev-setup  - Install/upgrade linting tools (ruff)"
	@echo "  make lint       - Lint all Python files recursively"
	@echo "  make fix        - Lint and auto-fix issues (including removing unused imports)"
	@echo "  make imports    - Remove unused imports only"

# Install tooling locally into the active environment
# Note: Prefer using a virtual environment before running this
dev-setup:
	python3 -m pip install --upgrade ruff

# Lint recursively (no changes written)
lint:
	python3 -m ruff check $(PY_DIRS)

# Auto-fix issues, including removing unused imports
fix:
	python3 -m ruff check --fix $(PY_DIRS)

# Remove only unused imports (F401) without touching other rules
imports:
	python3 -m ruff check --select F401 --fix $(PY_DIRS)
