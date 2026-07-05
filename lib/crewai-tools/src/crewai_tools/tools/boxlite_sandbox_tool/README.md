# BoxLite Sandbox Tools

Run shell commands, execute Python, and manage files inside a [BoxLite](https://boxlite.ai/) micro-VM. BoxLite boots an OCI image inside a hardware-isolated micro-VM **on the local host**, so — unlike the E2B and Daytona sandbox tools — there is no API key, account, or cloud round-trip. It is a self-hosted option for agent-driven code execution with a VM-strength isolation boundary.

Three tools are provided so you can pick what the agent actually needs:

- **`BoxLiteExecTool`** — run a shell command, returning exit code, stdout, and stderr.
- **`BoxLitePythonTool`** — run a block of Python (`python3`) and return its stdout, stderr, and exit code.
- **`BoxLiteFileTool`** — read / write / append / list / delete / mkdir / info / exists.

## Installation

```shell
uv add "crewai-tools[boxlite]"
# or
pip install "crewai-tools[boxlite]"
```

No API key is required. The host must support micro-VMs:

| Platform | Requirement |
| --- | --- |
| macOS 12+ | Apple Silicon (`Hypervisor.framework`) |
| Linux | KVM enabled (`/dev/kvm` accessible) |

## Box lifecycle

All three tools share the same lifecycle controls from `BoxLiteBaseTool`:

| Mode | When the box is created | When it is removed |
| --- | --- | --- |
| **Ephemeral** (default, `persistent=False`) | On every `_run` call | At the end of that same call |
| **Persistent** (`persistent=True`) | Lazily on first use | At process exit (via `atexit`), or manually via `tool.close()` |

Ephemeral mode is the safe default: nothing leaks if the agent forgets to clean up. Use persistent mode when you want filesystem state or installed packages to carry across steps — typical when pairing `BoxLiteFileTool` with `BoxLiteExecTool`.

Configure the box with `image` (default `python:slim`), `cpus`, and `memory_mib`.

## Examples

### One-shot Python execution (ephemeral)

```python
from crewai_tools import BoxLitePythonTool

tool = BoxLitePythonTool()
result = tool.run(code="print(sum(range(10)))")
# {"exit_code": 0, "stdout": "45\n", "stderr": ""}
```

### Multi-step shell session (persistent)

```python
from crewai_tools import BoxLiteExecTool, BoxLiteFileTool

exec_tool = BoxLiteExecTool(persistent=True, image="python:slim")
file_tool = BoxLiteFileTool(persistent=True, image="python:slim")

# Each tool keeps its own persistent box for the life of the process.
file_tool.run(action="write", path="/tmp/data.txt", content="1\n2\n3\n")
exec_tool.run(command="wc -l < /tmp/data.txt")
```

### Give the tools to an agent

```python
from crewai import Agent
from crewai_tools import BoxLiteExecTool, BoxLitePythonTool

analyst = Agent(
    role="Data Analyst",
    goal="Run code safely in an isolated micro-VM",
    backstory="Executes untrusted, model-generated code without touching the host.",
    tools=[BoxLitePythonTool(), BoxLiteExecTool()],
)
```

## Notes and differences from E2B/Daytona

- **Local only.** Boxes run on the host; there is no remote/attach-by-id mode, because BoxLite's synchronous API does not expose one with the rich exec (cwd/env/timeout) these tools use.
- **Text results only.** `BoxLitePythonTool` returns stdout/stderr/exit code; there is no Jupyter kernel, so rich results (charts, dataframes) are not returned.
- **Files over shell.** `BoxLiteFileTool` implements filesystem operations via the box's shell (content is transferred as base64), since BoxLite exposes command execution rather than a files API.
