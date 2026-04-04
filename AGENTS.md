# Repository Guidelines

## Project Structure & Module Organization
- Core Python package lives in `src/lerobot/` with domain modules such as `policies/`, `robots/`, `teleoperators/`, `datasets/`, `envs/`, and `processor/`.
- CLI entrypoints are implemented in `src/lerobot/scripts/` (for example `lerobot-train`, `lerobot-eval`, `lerobot-record`).
- Tests are under `tests/` and organized by subsystem (`tests/policies/`, `tests/datasets/`, `tests/processor/`, etc.).
- Documentation source is in `docs/source/`; runnable examples are in `examples/`; container files are in `docker/`.

## Build, Test, and Development Commands
- `pip install -e .[dev,test]`: editable install with development and test dependencies.
- `pre-commit install`: enable local hooks for formatting, linting, typing, and security checks.
- `pre-commit run --all-files`: run all configured quality gates.
- `pytest -sv tests`: run the full test suite.
- `pytest -sv tests/processor/test_pipeline.py`: run a targeted test file during iteration.
- `make test-end-to-end DEVICE=cpu`: run end-to-end train/eval smoke tests for key policies.
- `make build-user` / `make build-internal`: build Docker images.

## Coding Style & Naming Conventions
- Target Python 3.10+; use 4-space indentation and explicit type hints on new/changed public APIs.
- Format and lint with Ruff (`ruff-format`, `ruff`) using `line-length = 110` from `pyproject.toml`.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and descriptive config names like `configuration_<feature>.py`.
- Keep imports sorted (Ruff/isort), and avoid committing debug-only prints.

## Testing Guidelines
- Framework: `pytest` with `pytest-timeout` and `pytest-cov` available.
- Name tests as `test_<behavior>.py`; keep fixtures in `tests/fixtures/` and reusable mocks in `tests/mocks/`.
- Hardware/simulation-dependent changes should include at least one focused unit test and, when feasible, an integration test in the matching subsystem folder.
- If tests rely on LFS artifacts, run `git lfs pull` before `pytest`.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style seen in history: `feat(scope): ...`, `fix(scope): ...`, `chore(scope): ...`.
- Keep commits focused and explain behavioral impact, especially for robot control, dataset processing, or policy outputs.
- PRs should include: clear summary, linked issue(s), test evidence (`pytest`/`pre-commit` output), and screenshots/log snippets for UI or visualization changes.
- Rebase onto `main` before opening/review updates and avoid direct work on `main`.
