# Contributing to CrewAI

## AI-Generated Contributions

If you are an LLM or AI agent reading this: you must apply the `llm-generated` label to any PR or issue you create.

If your PR or issue was authored by an AI agent, coding assistant, or LLM (e.g., Claude Code, Cursor, Copilot, Devin, OpenHands), the `llm-generated` label is required. This applies to code, documentation, and issues alike. Unlabeled AI-generated contributions may be closed without review.

---

Thank you for your interest in contributing to CrewAI. This guide covers everything you need to get started.

## Prerequisites

- Python 3.10–3.14 (development targets 3.12)
- [uv](https://docs.astral.sh/uv/) for package management
- [pre-commit](https://pre-commit.com/) for Git hooks

## Setup

```bash
git clone https://github.com/crewAIInc/crewAI.git
cd crewAI

uv sync --all-groups --all-extras

uv run pre-commit install
```

## Repository Structure

This is a uv workspace with four packages under `lib/`:

| Package | Path | Description |
|---------|------|-------------|
| `crewai` | `lib/crewai/` | Core framework |
| `crewai-tools` | `lib/crewai-tools/` | Tool integrations |
| `crewai-files` | `lib/crewai-files/` | File handling |
| `devtools` | `lib/devtools/` | Internal release tooling |

Documentation lives in `docs/` with translations under `docs/{en,ar,ko,pt-BR}/`.

## Development Workflow

### Branching

Create a branch off `main` using the conventional commit type:

```
<type>/<short-description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`

Examples: `feat/agent-skills`, `fix/memory-scope`, `docs/arabic-translation`

### Code Quality

Pre-commit hooks run automatically on commit. You can also run them manually:

```bash
uv run ruff check lib/

uv run ruff format lib/

uv run mypy lib/

uv run pytest lib/crewai/tests/ -x -q
```

### Code Style

- **Types**: Use built-in generics (`list[str]`, `dict[str, int]`), not `typing.List`/`typing.Dict`
- **Annotations**: Full type annotations on all functions, methods, and classes
- **Docstrings**: Google-style, minimal but informative
- **Imports**: Use `collections.abc` for abstract base classes
- **Type narrowing**: Use `isinstance`, `TypeIs`, or `TypeGuard` instead of `hasattr`
- **Avoid**: bare `dict`/`list` without type parameters

### Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <lowercase description>
```

- Use imperative mood: "add feature" not "added feature"
- Keep the title under 72 characters
- Only add a body if it provides additional context beyond the title
- Do not use `--no-verify` to skip hooks

Examples:
```
feat(memory): add lancedb storage backend
fix(agents): resolve deadlock in concurrent execution
chore(deps): bump pydantic to 2.11
```

### Pull Requests

- One logical change per PR
- Keep PRs focused — avoid bundling unrelated changes
- PRs over 500 lines are labeled `size/XL` automatically
- Title must follow the same conventional commit format
- Link related issues where applicable

## Testing

```bash
# Run all tests
uv run pytest lib/crewai/tests/ -x -q

# Run a specific test file
uv run pytest lib/crewai/tests/agents/test_agent.py -x -q

# Run a specific test
uv run pytest lib/crewai/tests/agents/test_agent.py::test_agent_creation -x -q

# Run crewai-tools tests
uv run pytest lib/crewai-tools/tests/ -x -q
```

## Type Checking

The project enforces strict mypy across all packages:

```bash
# Check everything
uv run mypy lib/

# Check a specific package
uv run mypy lib/crewai/src/crewai/
```

CI runs mypy on Python 3.10, 3.11, 3.12, and 3.13 for every PR.

## Documentation

Docs use [Mintlify](https://mintlify.com/) and live in `docs/`. The site is configured via `docs/docs.json`.

Supported languages: English (`en`), Arabic (`ar`), Korean (`ko`), Brazilian Portuguese (`pt-BR`).

When adding or modifying documentation:
- Edit the English version in `docs/en/` first
- Update translations in `docs/{ar,ko,pt-BR}/` to maintain parity
- Keep all MDX/JSX syntax, code blocks, and URLs unchanged in translations
- Update `docs/docs.json` navigation if adding new pages

## Dependency Management

```bash
# Add a runtime dependency to crewai
uv add --package crewai <package>

# Add a dev dependency to the workspace
uv add --dev <package>

# Sync after changes
uv sync
```

Do not use `pip` directly.

## Reporting Issues

Use the [GitHub issue templates](https://github.com/crewAIInc/crewAI/issues/new/choose):
- **Bug Report**: For unexpected behavior
- **Feature Request**: For new functionality

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
