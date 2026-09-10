# LLMSandboxTool

Runs agent-authored code in an isolated container on infrastructure you already
operate, using [llm-sandbox](https://github.com/vndee/llm-sandbox).

Unlike `E2BPythonTool` and the Daytona sandbox tools, there is no API key and no
per-execution cost — code runs on your Docker, Podman or Kubernetes and never
leaves your machines. Supports Python, JavaScript, Java, C++, Go, R and Ruby.

## Installation

```bash
uv add crewai-tools --extra llm-sandbox
```

Requires a container runtime; Docker by default.

## Usage

```python
from crewai import Agent
from crewai_tools import LLMSandboxTool

agent = Agent(
    role="Data Analyst",
    goal="Answer quantitative questions by writing and running code",
    backstory="You prefer computing an answer over estimating one.",
    tools=[LLMSandboxTool()],
)
```

Another language, backend, or a pre-baked image:

```python
LLMSandboxTool(lang="ruby")
LLMSandboxTool(backend="kubernetes")
LLMSandboxTool(image="my-registry/python-with-pandas:1.0")
```

## Security

`DEFAULT_RUNTIME_CONFIGS` applies unless you pass `runtime_configs`:

```python
{
    "network_mode": "none",
    "mem_limit": "512m",
    "pids_limit": 128,
    "cap_drop": ["ALL"],
    "cap_add": ["DAC_OVERRIDE"],
    "security_opt": ["no-new-privileges:true"],
}
```

Verified inside the container: `CapEff` is `0000000000000002` and outbound
connections fail.

- **`DAC_OVERRIDE` is kept deliberately.** llm-sandbox copies the source file
  into the container; without it that file is unreadable and every run fails.
- **`read_only: True` is unsupported.** Docker rejects the code copy against a
  read-only rootfs, tmpfs on the workdir or not.
- **No package-installation argument.** The default network isolation would
  block it, and letting a model choose arbitrary PyPI packages executes
  `setup.py` at install time. Use `image=` instead.

This is container isolation, not VM isolation: it inherits the threat model of
the chosen backend and makes no kernel-level guarantee. For deliberately
adversarial code, pair it with a hardened runtime such as gVisor or Kata
Containers.
