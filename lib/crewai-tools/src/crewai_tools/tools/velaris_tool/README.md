# Velaris Tools

## Description

Run code your agents write, in a box. These tools compile a
[Velaris](https://github.com/gowrishankar-infra/velaris-lang) program,
report what it can touch, and run it under an effect budget chosen by
you, the crew's author, not by the agent.

- **VelarisRunTool**: runs a program under an effect budget. `VelarisRunTool(allow=["io"])` means anything the agent writes can print and nothing else - not read a file, reach the network, or call Python - whatever the code claims. A refusal stops the program and the tool reports which effect was refused.
- **VelarisAuditTool**: tells the agent (and you) what a program can touch and how much of its promises were proven before running, in the versioned `velaris.audit/1` JSON format.
- **VelarisCardTool**: returns the Velaris language card (about 2,300 words), which an agent can read before writing Velaris.

The program runs in a separate, killable process bounded by `timeout`
(seconds, default 30) and `max_memory_mb` (default 512). A program that
never ends or eats memory is stopped, and the tool reports which limit
it hit. The memory cap is enforced on Linux and macOS; on Windows it is
recorded but not enforced. Both limits need velaris-lang 2.59.0 or
newer; with an older compiler the program runs unbounded.

Not a security boundary: `allow=["ffi"]` grants everything Python can
do. It is a real guard for the ordinary case of running a script a
model wrote.

## Installation

Install the crewai_tools package and the Velaris compiler:

```shell
pip install 'crewai[tools]' velaris-lang
```

or, with the optional extra:

```shell
uv add crewai-tools --extra velaris
```

`z3-solver` is optional. Without it, promises are checked while running
rather than proven beforehand, and the audit says so.

## Example

```python
from crewai import Agent
from crewai_tools import VelarisAuditTool, VelarisRunTool

analyst = Agent(
    role="Data analyst",
    goal="Total the expenses in the CSV and report the largest",
    tools=[VelarisAuditTool(), VelarisRunTool(allow=["io"])],
)
```

To let the program read files as well:

```python
VelarisRunTool(allow=["io", "fs"])
```

## Arguments

### VelarisRunTool

Constructor:

- `allow`: (Optional) The effects the program may perform, chosen from `io`, `fs`, `net`, `clock`, `rand` and `ffi`. Defaults to `["io"]`.
- `timeout`: (Optional) Seconds the program may run before it is stopped. Defaults to `30.0`.
- `max_memory_mb`: (Optional) Memory the program may use, in MB, before it is stopped. Defaults to `512`. Enforced on Linux and macOS.

Run:

- `source`: The Velaris program to run.
- `stdin`: (Optional) Text to feed the program on standard input.
- `args`: (Optional) Command-line arguments for the program.

### VelarisAuditTool

- `source`: The Velaris program to inspect.

### VelarisCardTool

Takes no arguments.
