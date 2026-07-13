# K8s Agent Sandbox Tools

Run shell commands, execute Python, and manage files inside a [Kubernetes Agent Sandbox](https://github.com/kubernetes-sigs/agent-sandbox). K8s Agent Sandbox enables easy management of isolated, stateful, singleton workloads, ideal for use cases like AI agent runtimes.

Three tools are provided so you can pick what the agent actually needs:

- **`K8sAgentSandboxExecTool`** — run a shell command.
- **`K8sAgentSandboxFileTool`** — read / write / list / delete files.
- **`K8sAgentSandboxPythonTool`** — run a Python code.

## Installation

```shell
uv add "crewai-tools[k8s_agent_sandbox]"
# or
pip install "crewai-tools[k8s_agent_sandbox]"
```

## Sandbox toolset

Instead of configuring sandbox-related settings for each of the tools separately, the `K8sAgentSandboxToolset` class must be used.

All tools that are meant to share the same sandbox settings and sandbox lifecycle has to be added to the same instance of the
`K8sAgentSandboxToolset` class.

### Toolset example

This example expects that you have a cluster with the Agent Sandbox installed and all remaining prerequisites like sandbox router, sandbox template and a sandbox warmpool deployed. Learn more [here](https://github.com/kubernetes-sigs/agent-sandbox/tree/main#installation).

```python
from crewai_tools import (
    K8sAgentSandboxToolSandboxSettings as SandboxSettings,
    K8sAgentSandboxToolset,
    K8sAgentSandboxExecTool,
    K8sAgentSandboxFileTool,
)

toolset = K8sAgentSandboxToolset.create(
    # Sandbox settings for all tools in this toolset.
    sandbox_settings=SandboxSettings(
        warmpool="my-warmpool",
        namespace="my-namespace",
    ),
)

# Adding the tools to the toolset.
exec_tool = K8sAgentSandboxExecTool(toolset=toolset)
file_tool = K8sAgentSandboxFileTool(toolset=toolset)
```

### Lifecycle modes

The `K8sAgentSandboxToolset` supports three sandbox lifecycle modes and they can be set on creation of the toolset instance
with the `K8sAgentSandboxToolset.create` factory method:

| Mode | When the sandbox is created | When it is killed |
| --- | --- | --- |
| **Ephemeral** (default, `persistent=False`) | On every `_run` call of the tool | At the end of that same call |
| **Persistent** (`persistent=True`) | Lazily on first use | At process exit (via `atexit`), or manually via `toolset.close()` |
| **Attach** (`claim_name="…"`) | Never — the tool attaches to an existing sandbox | Never — the tool will not kill a sandbox it did not create |

Ephemeral mode is the safe default: nothing leaks if the agent forgets to clean up, but since it has to create a new sandbox on each new run, the increased execution time is expected. Use persistent mode when you want filesystem state or installed packages to carry across steps.

K8s agent sandboxes also auto-expire after an idle timeout. Tune it via `sandbox_timeout` (seconds, default `300`).

### Sandbox client settings

By default the `K8sAgentSandboxToolset.create` creates a client in the "Local Tunnel" configuration. To specify another configurations,
the `client_settings` keyword argument has to be used.

The following example shows setting up the sandbox client to use the "Gateway" configuration (learn more about all configurations [here](https://github.com/kubernetes-sigs/agent-sandbox/tree/main/clients/python/agentic-sandbox-client#usage-examples)):

```python
from crewai_tools import (
    K8sAgentSandboxToolClientSettings as ClientSettings, # <- import this.
    K8sAgentSandboxToolSandboxSettings as SandboxSettings,
    K8sAgentSandboxToolset,
    K8sAgentSandboxExecTool,
    K8sAgentSandboxFileTool,
)
from k8s_agent_sandbox.models import SandboxGatewayConnectionConfig


toolset = K8sAgentSandboxToolset.create(
    sandbox_settings=SandboxSettings(
        warmpool="my-warmpool",
        namespace="my-namespace",
    ),
    client_settings=ClientSettings(
        connection_config=SandboxGatewayConnectionConfig(
            gateway_name="external-http-gateway",  # Name of the Gateway resource
        )
    )
)

```

## Examples

### Ephemeral sandboxes

```python
from crewai_tools import (
    K8sAgentSandboxToolSandboxSettings as SandboxSettings,
    K8sAgentSandboxToolset,
)

toolset = K8sAgentSandboxToolset.create(
    sandbox_settings=SandboxSettings(
        warmpool="my-warmpool",
        namespace="my-namespace",
    ),
)

exec_tool = K8sAgentSandboxExecTool(toolset=toolset)
file_tool = K8sAgentSandboxFileTool(toolset=toolset)
```

### Persistent sandboxes

```python
from crewai_tools import (
    K8sAgentSandboxToolSandboxSettings as SandboxSettings,
    K8sAgentSandboxToolset,
)

toolset = K8sAgentSandboxToolset.create(
    sandbox_settings=SandboxSettings(
        warmpool="my-warmpool",
        namespace="my-namespace",
    ),
    persistent=True # <-- Add this.
)

exec_tool = K8sAgentSandboxExecTool(toolset=toolset)
file_tool = K8sAgentSandboxFileTool(toolset=toolset)
```

### Attach to an existing sandbox

```python
from crewai_tools import (
    K8sAgentSandboxToolSandboxSettings as SandboxSettings,
    K8sAgentSandboxToolset,
)

toolset = K8sAgentSandboxToolset.create(
    sandbox_settings=SandboxSettings(
        warmpool="my-warmpool",
        namespace="my-namespace",
    ),
    claim_name="<claim_name>" # <-- Add this.
)

exec_tool = K8sAgentSandboxExecTool(toolset=toolset)
file_tool = K8sAgentSandboxFileTool(toolset=toolset)
```

## Tool arguments

### `K8sAgentSandboxExecTool`
- `command: str` — shell command to run.
- `timeout: float | None` — seconds.

### `K8sAgentSandboxFileTool`
- `action: "read" | "write" | "append" | "list" | "delete" | "mkdir" | "info" | "exists"`
- `path: str` — absolute path inside the sandbox.
- `content: str | None` — required for `append`; optional for `write`.
- `binary: bool` — if `True`, `content` is base64 on write / returned as base64 on read.

### `K8sAgentSandboxPythonTool`
- `code: str` — source to execute.
- `timeout: float | None` — seconds.


## Security considerations

These tools hand the LLM arbitrary shell, Python, and filesystem access inside sandbox. The threat model to keep in mind:

- **Prompt-injection is a code-execution vector.** If the agent ingests untrusted content (web pages, scraped documents, user-supplied files, emails, search results), a malicious instruction hidden in that content can coerce the agent into issuing commands to `K8sAgentSandboxExecTool` / `K8sAgentSandboxPythonTool`. Treat any pipeline that feeds untrusted text into an agent that also has these tools as equivalent to remote code execution — the LLM is the attacker's shell.
- **Ephemeral mode (the default) is the main blast-radius control.** A fresh sandbox is created per call and killed at the end, so injected commands cannot persist state, exfiltrate long-lived secrets, or build up tooling across turns. Leave `persistent=False` unless you have a concrete reason to change it.
- **Avoid this specific combination:**
  - untrusted content in the agent's context, **plus**
  - `persistent=True` or an explicit long-lived `claim_name`, **plus**
  - a large `sandbox_timeout`.


## Full Example

```python
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    K8sAgentSandboxToolSandboxSettings as SandboxSettings,
    K8sAgentSandboxToolset,
    K8sAgentSandboxPythonTool,
    K8sAgentSandboxExecTool,
    K8sAgentSandboxFileTool,
)

# Create your toolset
toolset = K8sAgentSandboxToolset.create(
    sandbox_settings=SandboxSettings(
        warmpool="your-warmpool",
        namespace="default",
    ),
    persistent=True,
)

# Instantiate your tools
exec_tool = K8sAgentSandboxExecTool(toolset=toolset)
file_tool = K8sAgentSandboxFileTool(toolset=toolset)
python_tool = K8sAgentSandboxPythonTool(toolset=toolset)

# Define the Agent
devops_agent = Agent(
    role='Kubernetes DevOps Engineer',
    goal='Execute commands in a secure sandbox to inspect the environment safely.',
    backstory=(
        "You are a senior DevOps engineer specializing in Kubernetes. "
        "You use isolated pod sandboxes to run scripts, check system details, "
        "and securely verify environments without risking the host system."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[exec_tool, file_tool, python_tool]
)

# Define the Task
inspect_task = Task(
    description=(
        "Use your K8s sandbox execution tool to find out what operating system "
        "the sandbox pod is running. Run a command like 'cat /etc/os-release' "
        "and summarize the results."
    ),
    expected_output="A brief summary of the sandbox pod's operating system based on the command output.",
    # expected_output="Only name of the Operating System.",
    agent=devops_agent,
)

# Assemble the Crew
k8s_crew = Crew(
    agents=[devops_agent],
    tasks=[inspect_task],
    process=Process.sequential
)

# Run the Crew
if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set.")

    print("Starting the CrewAI execution...")
    result = k8s_crew.kickoff()

    print("\n================================================")
    print("FINAL RESULT FROM THE AGENT:")
    print("================================================")
    print(result)
```

