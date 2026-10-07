.PHONY: help install dev upgrade models fix lint type-check test test-integration docs docs-serve check run chat bench report clean

# Quieten third-party chatter that has nothing to say about our code.
export HF_HUB_DISABLE_PROGRESS_BARS := 1
export TOKENIZERS_PARALLELISM := false
export TRANSFORMERS_VERBOSITY := error

PY := .venv/bin/python
UV_PY := --python $(PY)

help:
	@echo "Setup:              install | dev | upgrade | models"
	@echo "Run:                run | chat | bench | report"
	@echo "Dev (modify files): fix"
	@echo "Checks (read-only): lint | type-check | test | test-integration | docs | check"
	@echo "Cleanup:            clean"

# ── Setup ──────────────────────────────────────────────────────────────────────

.venv:
	uv venv --python 3.13 .venv

install: .venv
	uv pip install $(UV_PY) -e .

dev: .venv
	uv pip install $(UV_PY) -e ".[dev,docs,bench]"
	.venv/bin/pre-commit install

upgrade: .venv
	uv pip install $(UV_PY) --upgrade -e ".[dev,docs,bench]"

# Pull the Ollama LLM and download ASR/TTS/embedding/VAD weights into data/models + HF cache.
models:
	$(PY) -m voicemate models

# ── Run ────────────────────────────────────────────────────────────────────────

# Optional: make run PROFILE=gemma (profiles are defined in config/voicemate.toml [llm]).
PROFILE_ARG := $(if $(PROFILE),--profile $(PROFILE),)

run:
	$(PY) -m voicemate serve $(PROFILE_ARG)

chat:
	$(PY) -m voicemate chat $(PROFILE_ARG)

bench:
	$(PY) -m voicemate bench all

report:
	$(PY) -m voicemate report

# ── Dev helpers (modify files) ─────────────────────────────────────────────────

fix:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

# ── Checks (read-only — mirrors GitHub CI) ─────────────────────────────────────

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

type-check:
	.venv/bin/ty check voicemate --python $(PY)

test:
	.venv/bin/coverage run -m unittest discover -s tests
	.venv/bin/coverage report
	.venv/bin/coverage html
	.venv/bin/coverage xml -o coverage.xml

# Needs Ollama running and the models from `make models`.
test-integration:
	VOICEMATE_INTEGRATION=1 $(PY) -m unittest discover -s tests -v

docs:
	.venv/bin/sphinx-build -W -b html docs/ site/

docs-serve: docs
	$(PY) -m http.server 8000 --directory site

check: lint type-check test docs

# ── Misc ───────────────────────────────────────────────────────────────────────

clean:
	rm -rf coverage_html dist/ site/ .ruff_cache
	rm -f .coverage coverage.xml
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
