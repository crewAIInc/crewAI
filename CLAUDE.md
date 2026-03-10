# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrewAI is a standalone Python framework for orchestrating autonomous AI agents. It provides two complementary paradigms: **Crews** (autonomous agent teams) and **Flows** (event-driven workflows). This is a **UV workspace monorepo**.

## Repository Structure

```
lib/
├── crewai/          # Core framework (agents, tasks, crews, flows, memory, tools, LLMs)
├── crewai-tools/    # Pre-built tool library (70+ tools)
├── crewai-files/    # Multimodal file handling (cache, processing, uploading)
└── devtools/        # Internal dev utilities (version bumping)
```

Source code lives under `lib/<package>/src/` and tests under `lib/<package>/tests/`.

## Common Commands

```bash
# Install dependencies
uv lock && uv sync

# Run all tests (parallel by default via pytest-xdist)
uv run pytest

# Run a single test file
uv run pytest lib/crewai/tests/memory/test_unified_memory.py

# Run a single test
uv run pytest lib/crewai/tests/memory/test_unified_memory.py::test_function_name -x

# Run tests for a specific workspace member
uv run pytest lib/crewai/tests/
uv run pytest lib/crewai-tools/tests/
uv run pytest lib/crewai-files/tests/

# Linting and formatting (Ruff)
uv run ruff check lib/
uv run ruff format lib/

# Type checking (strict mypy)
uv run mypy lib/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

**Pytest defaults** (from pyproject.toml): `--tb=short -n auto --timeout=60 --dist=loadfile --block-network --import-mode=importlib`. Network is blocked in tests; use VCR cassettes for HTTP interactions.

## Deep Dive

For detailed architecture documentation on any subsystem, use `/deep-dive <subsystem>` (e.g. `/deep-dive memory`, `/deep-dive flow`). This pulls the relevant section from **[EXPANDED_CLAUDE.md](./EXPANDED_CLAUDE.md)**, which covers all major components, execution flows, data types, and integration patterns. To regenerate it after major changes, use `/update-docs`.

## Architecture

### Core modules (`lib/crewai/src/crewai/`)

- **`crew.py`** - `Crew` class: orchestrates agents executing tasks (sequential or hierarchical process)
- **`task.py`** - `Task` class: work units with description, expected output, assigned agent, guardrails
- **`agent/core.py`** - `Agent` class: autonomous entity with role/goal/backstory, LLM, tools, memory
- **`flow/flow.py`** - `Flow` class: event-driven workflows using `@start`, `@listen`, `@router` decorators
- **`llm.py`** + **`llms/`** - Provider-agnostic LLM abstraction with per-provider adapters (OpenAI, Gemini, Claude, Bedrock, etc.)
- **`memory/`** - Unified memory system (LanceDB-backed) with vector embeddings, encoding/recall flows, scope-based filtering
- **`tools/`** - Tool ecosystem: `BaseTool`, structured tools, MCP integration, memory tools
- **`events/`** - Central event bus for observability (agent, crew, flow, task, memory events)
- **`knowledge/`** - Knowledge base integration with multiple source types
- **`cli/`** - CLI for project scaffolding, deployment, and interactive crew chat
- **`utilities/`** - Shared helpers (prompt templates, schema utils, LLM utils, i18n, guardrails)

### Key patterns

- **Pydantic models** throughout for validation and type safety
- **Event-driven observability** via `events/event_bus.py` with sync/async handlers
- **Lazy loading** of heavy modules (Memory, EncodingFlow) via `__getattr__`
- **Pluggable storage** backends for memory (LanceDB default)
- **VCR cassettes** for recording/replaying HTTP interactions in tests
- **Translations** in `translations/en.json` for all agent-facing prompts

## Code Standards

- **Python 3.10+**, use modern syntax (`X | Y` unions, `collections.abc`, f-strings)
- **Ruff** for linting and formatting (E501 line length ignored)
- **mypy strict** mode: all functions need type annotations
- **Google-style docstrings**
- **No relative imports** (`ban-relative-imports = "all"` in Ruff config)
- **Commitizen** commit message format enforced via pre-commit
- Tests allow `assert`, unnecessary assignments, and hardcoded passwords (`S101`, `RET504`, `S105`, `S106` suppressed)
