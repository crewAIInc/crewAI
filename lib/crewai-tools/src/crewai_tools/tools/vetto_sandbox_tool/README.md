# Vetto Sandbox Tools for CrewAI

The Vetto Sandbox toolset provides unprivileged, containerless process isolation for CrewAI agents executing shell commands and Python code.

It interfaces with the `vetto` kernel sandbox runtime (leveraging Linux Landlock LSM ABI 1-6, namespaces, cgroups v2, macOS Seatbelt, and Windows LPAC) without requiring Docker or root privileges.

## Tools Included

- `VettoExecTool`: Executes shell commands inside the sandbox with configurable filesystem boundaries, network modes (`off`, `allowlist`, `host`), timeouts, and memory ceilings.
- `VettoPythonTool`: Executes isolated Python scripts or code snippets.
- `VettoFileTool`: Reads, writes, appends, and manages files strictly within the workspace directory root, preventing path traversal.

## Installation

Install Vetto CLI:
```bash
# Via cargo
cargo install vetto

# Or via npm
npm install -g @shledery/vetto

# Or via Homebrew
brew install shleder/tap/vetto
```

## Usage Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import VettoExecTool, VettoPythonTool, VettoFileTool

workspace_dir = "./sandbox_workspace"

exec_tool = VettoExecTool(
    working_dir=workspace_dir,
    net="off",
    timeout=60,
    memory_limit="512MB",
)

python_tool = VettoPythonTool(
    working_dir=workspace_dir,
    net="off",
    timeout=30,
)

file_tool = VettoFileTool(
    working_dir=workspace_dir,
)

coding_agent = Agent(
    role="Sandboxed Code Engineer",
    goal="Safely run analysis scripts without host modification",
    backstory="You execute code inside a secure local sandbox runtime.",
    tools=[exec_tool, python_tool, file_tool],
)
```
