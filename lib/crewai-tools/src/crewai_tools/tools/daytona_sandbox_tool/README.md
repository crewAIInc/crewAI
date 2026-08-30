# Daytona Sandbox Tools

Run shell commands, execute Python, and manage files inside a [Daytona](https://www.daytona.io/) sandbox. Daytona provides isolated, ephemeral compute environments suitable for agent-driven code execution.

Three tools are provided so you can pick what the agent actually needs:

- **`DaytonaExecTool`** — run a shell command (`sandbox.process.exec`).
- **`DaytonaPythonTool`** — run a Python script (`sandbox.process.code_run`).
- **`DaytonaFileTool`** — read / write / list / delete files (`sandbox.fs.*`).

## Installation

```shell
uv add "crewai-tools[daytona]"
# or
pip install "crewai-tools[daytona]"
```

Set the API key:

```shell
export DAYTONA_API_KEY="..."
```

`DAYTONA_API_URL` and `DAYTONA_TARGET` are also respected if set.

## Sandbox lifecycle

All three tools share the same lifecycle controls from `DaytonaBaseTool`:

| Mode | When the sandbox is created | When it is deleted |
| --- | --- | --- |
| **Ephemeral** (default, `persistent=False`) | On every `_run` call | At the end of that same call |
| **Persistent** (`persistent=True`) | Lazily on first use | At process exit (via `atexit`), or manually via `tool.close()` |
| **Attach** (`sandbox_id="…"`) | Never — the tool attaches to an existing sandbox | Never — the tool will not delete a sandbox it did not create |

Ephemeral mode is the safe default: nothing leaks if the agent forgets to clean up. Use persistent mode when you want filesystem state or installed packages to carry across steps — this is typical when pairing `DaytonaFileTool` with `DaytonaExecTool`.

## Examples

### One-shot Python execution (ephemeral)

```python
from crewai_tools import DaytonaPythonTool

tool = DaytonaPythonTool()
result = tool.run(code="print(sum(range(10)))")
```

### Multi-step shell session (persistent)

```python
from crewai_tools import DaytonaExecTool, DaytonaFileTool

exec_tool = DaytonaExecTool(persistent=True)
file_tool = DaytonaFileTool(persistent=True)

# Agent writes a script, then runs it — but each tool keeps its OWN persistent
# sandbox. To share the *same* sandbox across two tools, create and use the
# first tool, then read its `active_sandbox_id` and pass it to the second:
#   exec_tool.run(command="pip install httpx")
#   file_tool = DaytonaFileTool(sandbox_id=exec_tool.active_sandbox_id)
```

### Attach to an existing sandbox

```python
from crewai_tools import DaytonaExecTool

tool = DaytonaExecTool(sandbox_id="my-long-lived-sandbox")
```

### Custom create params

Pass Daytona's `CreateSandboxFromSnapshotParams` kwargs via `create_params`:

```python
tool = DaytonaExecTool(
    persistent=True,
    create_params={
        "language": "python",
        "env_vars": {"MY_FLAG": "1"},
        "labels": {"owner": "crewai-agent"},
    },
)
```

## Tool arguments

### `DaytonaExecTool`
- `command: str` — shell command to run.
- `cwd: str | None` — working directory.
- `env: dict[str, str] | None` — extra env vars for this command.
- `timeout: int | None` — seconds.

### `DaytonaPythonTool`
- `code: str` — Python source to execute.
- `argv: list[str] | None` — argv forwarded via `CodeRunParams`.
- `env: dict[str, str] | None` — env vars forwarded via `CodeRunParams`.
- `timeout: int | None` — seconds.

### `DaytonaFileTool`
- `action`: one of `read`, `write`, `append`, `list`, `delete`, `mkdir`, `info`, `exists`, `move`, `find`, `search`, `chmod`, `replace`.
- `path: str | None` — absolute path inside the sandbox. Required for all actions except `replace`.
- `content: str | None` — required for `append`; optional for `write`.
- `binary: bool` — if `True`, `content` is base64 on write / returned as base64 on read.
- `recursive: bool` — for `delete`, removes directories recursively.
- `mode: str | None` — for `mkdir` (defaults to `"0755"`) or for `chmod` (e.g. `"755"`).
- `destination: str | None` — required for `move`.
- `pattern: str | None` — required for `find` (content grep), `search` (filename glob), and `replace`.
- `replacement: str | None` — required for `replace`.
- `paths: list[str] | None` — required for `replace`; list of files to operate on.
- `owner: str | None` / `group: str | None` — for `chmod`. Pass at least one of `mode`, `owner`, or `group`.
