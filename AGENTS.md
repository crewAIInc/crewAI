# AGENTS.md -- crewAI Repository Contributor Guide for AI Coding Assistants

> Repo-local instructions for AI coding assistants (Claude Code, Cursor,
> Windsurf, GitHub Copilot, Devin, Codex, etc.) contributing to the **crewAI
> monorepo**. Follow these deterministic validation steps before opening or
> updating any pull request.

---

## Repository Layout

```
crewAI/                         # workspace root (uv workspace)
  pyproject.toml                # workspace-level config (ruff, mypy, pytest, uv)
  conftest.py                   # shared pytest fixtures (VCR, event cleanup, env)
  uv.lock
  lib/
    crewai/                     # core framework package
      src/crewai/               # source code
      tests/                    # unit & integration tests
      pyproject.toml
    crewai-tools/               # tool integrations package
      src/crewai_tools/         # source code
      tests/                    # unit & integration tests
      pyproject.toml
    crewai-files/               # file-processing package
      src/crewai_files/
      tests/
      pyproject.toml
    devtools/                   # internal developer tooling (lib/devtools)
      src/
      pyproject.toml
  docs/                         # documentation site content
  .github/workflows/            # CI definitions (lint, tests, type-check)
```

Key conventions:
- **uv workspace** -- all packages are managed via `uv` with workspace members
  declared in the root `pyproject.toml`.
- Source lives under `lib/<package>/src/`, tests under `lib/<package>/tests/`.
- Shared test configuration (fixtures, VCR cassette filtering) is in the
  root `conftest.py`.

---

## Environment Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install all workspace dependencies (including dev tools)
uv sync --all-groups --all-extras

# 3. Verify the virtual environment
uv run python -c "import crewai; print(crewai.__version__)"
```

Python version requirement: `>=3.10, <3.14` (see root `pyproject.toml`).

---

## Deterministic Validation Workflow

Run these checks **in order** before pushing commits. Every command must exit
with code 0.

### 1. Lint (ruff)

```bash
# Check for lint errors on changed files (mirrors CI)
uv run ruff check --config pyproject.toml <files>

# Or check all source files
uv run ruff check --config pyproject.toml lib/
```

The CI job runs ruff only on changed Python files, excluding
`lib/crewai/src/crewai/cli/templates/` and `*/tests/` directories.

Configuration: root `pyproject.toml` `[tool.ruff]` section.

### 2. Format (ruff)

```bash
# Check formatting (dry-run)
uv run ruff format --check --config pyproject.toml lib/

# Auto-fix formatting
uv run ruff format --config pyproject.toml <files>
```

### 3. Type Check (mypy)

```bash
# Run mypy on changed source files (excluding tests and templates)
uv run mypy --config-file pyproject.toml <files>
```

CI runs mypy only on changed files under `lib/*/src/` (not tests).
Configuration: root `pyproject.toml` `[tool.mypy]` section with
`strict = true`.

### 4. Tests (pytest)

```bash
# Run the full crewai test suite
cd lib/crewai && uv run pytest -vv --maxfail=3

# Run the crewai-tools test suite
cd lib/crewai-tools && uv run pytest -vv --maxfail=3

# Run the crewai-files test suite
cd lib/crewai-files && uv run pytest -vv --maxfail=3

# Run a single test file
cd lib/crewai && uv run pytest tests/path/to/test_file.py -vv

# Run a single test
cd lib/crewai && uv run pytest tests/path/to/test_file.py::test_function -vv
```

Test configuration (root `pyproject.toml` `[tool.pytest.ini_options]`):
- `asyncio_mode = "strict"` -- async tests require explicit
  `@pytest.mark.asyncio`.
- `--block-network` -- network calls are blocked by default; use VCR
  cassettes for HTTP interactions.
- `--timeout=60` -- per-test timeout of 60 seconds.
- `-n auto` -- parallel execution via pytest-xdist.
- `--dist=loadfile` -- tests from the same file run in the same worker.

### 5. Pre-commit (optional local check)

```bash
uv run pre-commit run --all-files
```

Runs ruff lint, ruff format, mypy, uv-lock consistency, and commitizen
checks.

---

## CI Workflows (GitHub Actions)

Pull requests trigger these workflows automatically:

| Workflow | File | What it checks |
|----------|------|----------------|
| **Lint** | `.github/workflows/linter.yml` | `ruff check` on changed `.py` files |
| **Tests** | `.github/workflows/tests.yml` | `pytest` across Python 3.10--3.13, split into 8 groups |
| **Type Check** | `.github/workflows/type-checker.yml` | `mypy` on changed source files across Python 3.10--3.13 |

All three must pass before merge.

---

## Safety Boundaries

- **Never commit secrets.** API keys, tokens, and credentials belong in
  `.env` (git-ignored) or environment variables. The `.env.test` file
  provides safe placeholder values for the test suite.
- **Never modify VCR cassettes by hand.** Cassettes are auto-generated YAML
  recordings of HTTP interactions. To re-record, set
  `PYTEST_VCR_RECORD_MODE=all` and run the relevant tests. Cassette paths
  mirror the test directory structure under `tests/cassettes/`.
- **Never modify tests just to make them pass.** If a test fails, fix the
  source code or report the issue. Do not weaken assertions, add
  `pytest.skip`, or catch exceptions to hide failures.
- **Respect `--block-network`.** Tests run with network blocking enabled.
  All external HTTP calls must use VCR cassettes. Do not disable this.
- **Do not add new dependencies without justification.** The workspace pins
  dependencies tightly. Any new dependency needs explicit rationale and must
  be added via `uv add`, not by editing `pyproject.toml` by hand.
- **Do not push directly to `main`.** All changes go through pull requests.

---

## Commit Convention

This repository uses [Commitizen](https://commitizen-tools.github.io/commitizen/)
with [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

# Examples:
feat(agent): add inject_date parameter
fix(llm): handle empty response from Bedrock
docs: update AGENTS.md validation commands
test(crew): add hierarchical process edge case
refactor(flow): simplify state persistence logic
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`, `revert`.

The `commitizen` pre-commit hook validates commit messages on push.

---

## Testing Conventions

- **Test file naming:** `test_*.py` in the appropriate `tests/` directory.
- **Test function naming:** `test_*` prefix.
- **Test class naming:** `Test*` prefix.
- **Fixtures:** Shared fixtures live in the root `conftest.py`. Package-
  specific fixtures go in `lib/<package>/tests/conftest.py`.
- **VCR cassettes:** Use `@pytest.mark.vcr` for tests that make HTTP calls.
  Cassettes are stored in `tests/cassettes/` mirroring the test directory
  structure.
- **Async tests:** Must be decorated with `@pytest.mark.asyncio` (strict
  mode).
- **Telemetry tests:** Mark with `@pytest.mark.telemetry` to opt out of
  the default telemetry mocking.

---

## Code Style

- **Linter/formatter:** ruff (config in root `pyproject.toml`).
- **Type checker:** mypy with `strict = true`.
- **Imports:** Absolute imports only (`ban-relative-imports = "all"`).
  Sorted by isort rules. No unused imports.
- **Line length:** No hard limit enforced (`E501` ignored), but keep lines
  reasonable.
- **Print statements:** Disallowed in source code (ruff rule `T`). Use
  proper logging.
- **Security:** Bandit rules enforced (ruff rule `S`). `assert` statements
  are allowed in test files only.
- **Type annotations:** Required on all function signatures in source code
  (mypy `disallow_untyped_defs = true`). Use `X | Y` union syntax (not
  `Optional` or `Union`), `collections.abc` types, and built-in generics
  (`list`, `dict` instead of `typing.List`, `typing.Dict`).
