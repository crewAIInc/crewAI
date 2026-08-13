# TaskMarket Tool

`TaskMarketTool` lets a CrewAI agent discover public TaskMarket work, inspect a task and its available submissions, and prepare a requester task without performing an external write. The integration is designed around TaskMarket’s documented task endpoints and the first-party TaskMarket command-line client.[1] [2]

> **Safety boundary.** Discovery and drafting do not create external state. The only creation path requires a fresh exact confirmation phrase, an explicit maximum-spend value at least equal to the requested reward, and a separately configured first-party TaskMarket CLI. The tool does not accept or store wallet keys, seed phrases, passwords, tokens, or payment credentials.

## Capabilities

| Action | Effect | Safety behavior |
|---|---|---|
| `list_tasks` | Lists public tasks, with normalized USDC reward and deadline fields. | Performs a public `GET` request only. |
| `get_task` | Retrieves a single task, including its description. | Performs a public `GET` request only. |
| `list_submissions` | Retrieves the available submission response for a task. | Performs a public `GET` request only. |
| `draft_task` | Produces the exact task payload, Base-network context, UTC deadline estimate, and first-party CLI preview. | Performs **no** write or CLI invocation. |
| `create_task` | Delegates a reviewed task request to the local first-party TaskMarket CLI. | Requires `confirmation="CREATE_TASKMARKET_TASK"`, a sufficient `max_spend_usdc`, and an installed `taskmarket` CLI. It never retries an uncertain result. |

The tool intentionally does **not** claim tasks, submit work, accept or reject submissions, rate work, cancel work, or create private/password-protected tasks. These exclusions keep LLM-selected input on the least-privileged path.

## Installation

This contribution is currently a local implementation candidate and is **not yet merged upstream**. To run it from this CrewAI workspace, install the repository’s development environment and execute the focused test suite:

```bash
cd crewai
uv run --group dev --package crewai-tools pytest -q \
  lib/crewai-tools/tests/tools/taskmarket_tool/test_taskmarket_tool.py
```

To use the eventual upstream package in a CrewAI project, import the standard public export:

```python
from crewai_tools import TaskMarketTool

taskmarket = TaskMarketTool()
```

## Read-only discovery

The following call retrieves up to five open public tasks. It does not use a wallet or perform any marketplace write.

```python
result = taskmarket.run(
    action="list_tasks",
    status="open",
    limit=5,
    tags=["research"],
)
print(result)
```

Each task summary includes gross and net values normalized from TaskMarket’s micro-USDC fields, its mode, tags, expiration time, submission count, and a direct task URL.

## Draft before external creation

A requester should first generate and review a no-write draft. The draft includes the full description with a separately visible deliverables section, reward, duration, estimated UTC deadline, network, visibility, and the exact first-party CLI command preview.

```python
draft = taskmarket.run(
    action="draft_task",
    description="Review the public documentation and identify implementation gaps.",
    deliverables="One Markdown report with source URLs and a prioritized findings table.",
    reward_usdc="2.50",
    duration_hours=24,
    tags=["research", "documentation"],
    mode="bounty",
    task_visibility="public",
    submission_visibility="winner_only",
)
print(draft)
```

The draft does not invoke the CLI, create a task, or initiate any funding step.

## Explicitly authorized creation

TaskMarket’s creation endpoint is authorization- and x402-gated.[1] This tool therefore leaves task creation to the first-party CLI already configured by the end user. The caller must deliberately re-enter the exact confirmation phrase and provide a maximum-spend value that covers the requested reward. The tool passes structured argument-list values to `taskmarket task create`; it never passes a shell string or exposes secret material.

```python
created = taskmarket.run(
    action="create_task",
    description="Review the public documentation and identify implementation gaps.",
    deliverables="One Markdown report with source URLs and a prioritized findings table.",
    reward_usdc="2.50",
    duration_hours=24,
    tags=["research", "documentation"],
    max_spend_usdc="2.50",
    confirmation="CREATE_TASKMARKET_TASK",
)
print(created)
```

If the command times out or fails ambiguously, the tool returns an **unknown-settlement** status and does not retry. The user must inspect TaskMarket directly before deciding what to do next.

## Validation

The focused tests use mocks for network reads and CLI calls. They verify public-response formatting, draft-only behavior, required deliverables, confirmation and maximum-spend gates, CLI result handling, and no retry after a timeout.

```bash
cd crewai
uv run --group dev ruff check \
  lib/crewai-tools/src/crewai_tools/tools/taskmarket_tool \
  lib/crewai-tools/tests/tools/taskmarket_tool
uv run --group dev --package crewai-tools pytest -q \
  lib/crewai-tools/tests/tools/taskmarket_tool/test_taskmarket_tool.py
```

## References

[1]: https://api.taskmarket.dev/openapi.json "TaskMarket OpenAPI specification"
[2]: https://www.npmjs.com/package/@lucid-agents/taskmarket "TaskMarket first-party CLI package"
[3]: https://github.com/crewAIInc/crewAI "CrewAI official source repository"
