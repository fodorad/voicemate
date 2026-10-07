# Contributing

## Setup

```bash
git clone https://github.com/fodorad/voicemate
cd voicemate
make dev      # .venv with uv (Python 3.13) + dev/docs/bench extras + pre-commit hook
make models   # Ollama LLM, ASR, TTS voices and embedding model (several GB)
```

## Workflow

GitHub Flow: short-lived `feat/*`, `fix/*`, `docs/*` or `ci/*` branches, pull requests into
`main`, squash-merge. release-please turns the Conventional Commits on `main` into a release
PR; merging that PR tags the version.

The project follows the AI-native SDLC in [docs/sdlc](docs/sdlc): changes of intent go into
`intent.md`, design decisions and their measurements into `spec.md`.

## Test-driven development

Write the test first. Tests exercise real behaviour with the standard library `unittest`,
random arrays for shape checks and small deterministic implementations of the model
protocols in `tests/helpers.py` (no mocks or patches). `tests/` mirrors `voicemate/`.

- `make test`: unit tests with coverage (≥ 80 % enforced), no models needed.
- `make test-integration`: also runs the tests that need Ollama, the downloaded models,
  the network and macOS `say`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
`refactor:`, `test:`, `chore:`, `ci:`, `perf:`, `build:`. They drive the changelog.

## Docstrings

Google-style docstrings, including module-level attributes and constants (`#:` comments),
so the Sphinx API reference is complete.

## Before opening a PR

```bash
make check   # lint + type-check + test + docs
```
