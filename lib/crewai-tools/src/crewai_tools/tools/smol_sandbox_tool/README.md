# Smol Machines sandbox tools

Run CrewAI code and file tools inside a microVM, locally or in Smol Machines Cloud. `SmolExecTool` runs shell commands, `SmolPythonTool` runs Python, and `SmolFileTool` reads and writes guest files. The image defaults to `python:3.12-alpine`.

```sh
uv add 'crewai-tools[smol]'
```

For a local microVM, use the tools directly (Linux requires `/dev/kvm`; macOS requires Apple Silicon):

```python
from crewai import Agent
from crewai_tools import SmolExecTool

agent = Agent(
    role="Developer",
    goal="Implement and test a small feature",
    backstory="Build and verify changes in an isolated computer.",
    tools=[SmolExecTool()],
)
```

For Cloud, set `SMOL_CLOUD_TOKEN` on the worker and select it explicitly:

```python
from crewai_tools import SmolExecTool

shell = SmolExecTool(target="cloud", persistent=True)
try:
    print(shell.run(command="echo ready > /workspace/status"))
    print(shell.run(command="cat /workspace/status"))
finally:
    shell.close()
```

`target="local"` is the default, even if a Cloud token is present. Guest egress is enabled by default for workloads that need network access; set `network=False` to block it. Cached registry images can still boot without guest egress. For a cold local image pull with guest egress disabled, SmolVM 1.23.0 or newer fetches the image on the host; on older versions, enable guest networking or supply a local rootfs directory or `docker save` archive (`image="./image.tar"`). For restricted egress, configure the VM directly with the Smol SDK and attach to it by ID.

By default, each call creates and deletes a VM. With `persistent=True`, calls on one tool share a VM until `close()` or process exit. To use multiple tools with one VM, create it through the first persistent tool and pass `active_machine_id` to another tool as `machine_id` with the same target:

```python
from crewai_tools import SmolExecTool, SmolFileTool

shell = SmolExecTool(persistent=True)
try:
    shell.run(command="echo ready")
    files = SmolFileTool(machine_id=shell.active_machine_id)
    files.run(action="write", path="/workspace/job.txt", content="hello")
    print(shell.run(command="cat /workspace/job.txt"))
finally:
    shell.close()
```

Attached tools do not delete a VM owned by another tool. `SmolFileTool` accepts UTF-8 text or base64 data (`binary=True`). Run `SmolPythonTool().run(code="print(2 + 2)")` to execute Python in the same image.
