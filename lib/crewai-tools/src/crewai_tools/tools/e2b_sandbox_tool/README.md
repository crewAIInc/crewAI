# E2B Sandbox Tools

Run shell commands, execute Python, and manage files inside an [E2B](https://e2b.dev/) sandbox. E2B provides isolated, ephemeral VMs suitable for agent-driven code execution, with a Jupyter-style code interpreter for rich Python results.

Three tools are provided so you can pick what the agent actually needs:

- **`E2BExecTool`** — run a shell command (`sandbox.commands.run`).
- **`E2BPythonTool`** — run a Python cell in the E2B code interpreter (`sandbox.run_code`), returning stdout/stderr and rich results (charts, dataframes).
- **`E2BFileTool`** — read / write / list / delete files (`sandbox.files.*`).

## Installation

```shell
uv add "crewai-tools[e2b]"
# or
pip install "crewai-tools[e2b]"
```

Set the API key:

```shell
export E2B_API_KEY="..."
```

`E2B_DOMAIN` is also respected if set (for self-hosted or non-default deployments).

## Sandbox lifecycle

All three tools share the same lifecycle controls from `E2BBaseTool`:

| Mode | When the sandbox is created | When it is killed |
| --- | --- | --- |
| **Ephemeral** (default, `persistent=False`) | On every `_run` call | At the end of that same call |
| **Persistent** (`persistent=True`) | Lazily on first use | At process exit (via `atexit`), or manually via `tool.close()` |
| **Attach** (`sandbox_id="…"`) | Never — the tool attaches to an existing sandbox | Never — the tool will not kill a sandbox it did not create |

Ephemeral mode is the safe default: nothing leaks if the agent forgets to clean up. Use persistent mode when you want filesystem state or installed packages to carry across steps — this is typical when pairing `E2BFileTool` with `E2BExecTool`.

E2B sandboxes also auto-expire after an idle timeout. Tune it via `sandbox_timeout` (seconds, default `300`).

## Examples

### One-shot Python execution (ephemeral)

```python
from crewai_tools import E2BPythonTool

tool = E2BPythonTool()
result = tool.run(code="print(sum(range(10)))")
```

### Multi-step shell session (persistent)

```python
from crewai_tools import E2BExecTool, E2BFileTool

exec_tool = E2BExecTool(persistent=True)
file_tool = E2BFileTool(persistent=True)

# Each tool keeps its own persistent sandbox. If you need the *same* sandbox
# across two tools, create one tool, grab the sandbox id via
# `tool._persistent_sandbox.sandbox_id`, and pass it to the other via
# `sandbox_id=...`.
```

### Attach to an existing sandbox

```python
from crewai_tools import E2BExecTool

tool = E2BExecTool(sandbox_id="sbx_...")
```

### Custom create params

```python
tool = E2BExecTool(
    persistent=True,
    template="my-custom-template",
    sandbox_timeout=600,
    envs={"MY_FLAG": "1"},
    metadata={"owner": "crewai-agent"},
)
```

## Tool arguments

### `E2BExecTool`
- `command: str` — shell command to run.
- `cwd: str | None` — working directory.
- `envs: dict[str, str] | None` — extra env vars for this command.
- `timeout: float | None` — seconds.

### `E2BPythonTool`
- `code: str` — source to execute.
- `language: str | None` — override kernel language (default: Python).
- `envs: dict[str, str] | None` — env vars for the run.
- `timeout: float | None` — seconds.

### `E2BFileTool`
- `action: "read" | "write" | "append" | "list" | "delete" | "mkdir" | "info" | "exists"`
- `path: str` — absolute path inside the sandbox.
- `content: str | None` — required for `append`; optional for `write`.
- `binary: bool` — if `True`, `content` is base64 on write / returned as base64 on read.
- `depth: int` — for `list`, how many levels to recurse (default 1).

## Security considerations

These tools hand the LLM arbitrary shell, Python, and filesystem access inside a remote VM. The threat model to keep in mind:

- **Prompt-injection is a code-execution vector.** If the agent ingests untrusted content (web pages, scraped documents, user-supplied files, emails, search results), a malicious instruction hidden in that content can coerce the agent into issuing commands to `E2BExecTool` / `E2BPythonTool`. Treat any pipeline that feeds untrusted text into an agent that also has these tools as equivalent to remote code execution — the LLM is the attacker's shell.
- **Ephemeral mode (the default) is the main blast-radius control.** A fresh sandbox is created per call and killed at the end, so injected commands cannot persist state, exfiltrate long-lived secrets, or build up tooling across turns. Leave `persistent=False` unless you have a concrete reason to change it.
- **Avoid this specific combination:**
  - untrusted content in the agent's context, **plus**
  - `persistent=True` or an explicit long-lived `sandbox_id`, **plus**
  - a large `sandbox_timeout` or credentials/secrets seeded into the sandbox via `envs`.

  That stack lets a single injection pivot into a long-running, credentialed shell that survives across turns. If you must run persistently, also keep `sandbox_timeout` short, scope `envs` to the minimum the task needs, and don't feed the same agent untrusted input.
- **Don't mount production credentials.** Anything you put into `envs`, `metadata`, or files written to the sandbox is reachable from the LLM. Use per-task scoped keys, not your personal API tokens.
- **E2B's VM isolation is the final backstop**, not a license to relax the above — isolation prevents escape to the host, but everything the sandbox can reach (the public internet, any service whose token you dropped in) is still fair game for an injected command.
