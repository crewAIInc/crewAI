# Agent Instructions for CrewAI OSS

CrewAI is a Python based framework for building AI agents and agentic systems.
Follow these guidelines when contributing:

## AI-Generated Contributions

If you are an LLM or AI agent: you **must** apply the `llm-generated` label to any PR
or issue you create. Unlabeled AI-generated contributions may be closed without
review. This applies to code, documentation, and issues alike.

## Setup

This repository is a [uv](https://docs.astral.sh/uv/) workspace. Packages live under
`lib/`: `crewai` (core), `crewai-core`, `crewai-tools`, `crewai-files`, `cli`, and
`devtools`. Python 3.10–3.14 is supported; development targets 3.12.

```bash
uv sync --all-groups --all-extras
uv run pre-commit install
```

## Checks

Run these before opening a PR. All of them are enforced in CI.

```bash
uv run pytest lib/crewai/tests/ -x -q   # tests for the core package
uv run ruff check lib/                  # lint
uv run ruff format lib/                 # format (CI runs it with --check)
uv run mypy lib/                        # type check, strict mode
```

## Gotchas

- Tests run with `--block-network`. HTTP is replayed from VCR cassettes — a test that
  reaches the live network will fail.
- PR titles are CI-enforced Conventional Commits: `<type>(<scope>): description`,
  lowercase, no trailing period. Allowed types: `feat`, `fix`, `refactor`, `perf`,
  `test`, `docs`, `chore`, `ci`, `style`, `revert`.
- Branch off `main` as `<type>/<short-description>`, e.g. `fix/memory-scope`.

See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md) for the full guide.

## Key Guidelines

1. Follow Python best practices and idiomatic patterns.
2. Maintain existing code structure and organization.
3. Write unit tests for new functionality focusing on behaivor and not
   implementation.
4. Document public APIs and complex logic.
5. Suggest changes to the `docs/` folder when appropriate
6. Follow software principles such as DRY and YAGNI.
7. Keep diffs as minimal as possible.

## Changing Docs

1. Edit MDX under `docs/edge/en/*` and reference it from `docs/docs.json` if
   needed.
2. Do not modify files under `docs/v*/`. Those are frozen release snapshots
   managed by devtools.
3. Do not delete or rename files under `docs/images/` as frozen snapshots
   may reference them.
4. If you want to preview your changes locally, use `cd docs && mintlify dev`.
   To check for broken links, run `cd docs && mintlify broken-links`.
