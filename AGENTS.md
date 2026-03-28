# AGENTS Instructions

This file provides machine-readable contributor guidance for CrewAI.

## Scope

- Keep changes focused to one behavior contract per PR.
- Prefer deterministic behavior and fail-closed defaults for tool/permission boundaries.
- Do not bundle unrelated refactors.

## Setup

- Python: `>=3.10,<3.14`
- Package manager: `uv`
- Create environment from repo root:
  - `uv venv`
  - `uv sync`

## Validation (repo-local baseline)

Run these from `lib/crewai/` unless otherwise noted.

1. Format/lint before tests:
- `uv run ruff check .`
- `uv run ruff format .`

2. Type checks:
- `uvx mypy src`

3. Tests:
- `uv run pytest .`

4. Build sanity (if packaging changes):
- `uv build`

## Engineering Guardrails

- Preserve deterministic crew/task execution order; add regression tests for ordering and retry semantics.
- Keep tool execution and permission behavior explicit; prefer hard failure over permissive fallback on ambiguous safety state.
- For concurrency changes, include tests that verify stable ordering or explicit ordering metadata.

## PR Hygiene

- PR description must include: Problem, What changed, Validation commands + results.
- Keep branch scope aligned to issue scope.
- If scope changes, update PR title/body immediately.

