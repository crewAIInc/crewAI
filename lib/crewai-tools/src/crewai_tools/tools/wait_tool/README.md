# WaitTool

The **WaitTool** pauses execution for a given number of seconds. Use it when an agent needs to
let time pass before re-checking a long-running job — a sandbox build, a deployment, a batch
import, an async API call.

No API key, no dependencies beyond the standard library.

## Arguments

| Argument  | Type    | Required | Description                                                                  |
| --------- | ------- | -------- | ---------------------------------------------------------------------------- |
| `seconds` | `float` | ✅        | How many seconds to wait. Must be zero or greater.                           |
| `reason`  | `str`   | ❌        | Optional note on what is being waited for. Echoed back in the tool's result. |

## Initialization Parameters

| Parameter     | Type    | Default | Description                                                                            |
| ------------- | ------- | ------- | -------------------------------------------------------------------------------------- |
| `max_seconds` | `float` | `300`   | Upper bound for a single wait. Longer requests are capped to this value, not rejected. |

## Usage Example

```python
from crewai import Agent
from crewai.tools import tool
from crewai_tools import WaitTool

wait_tool = WaitTool()


@tool("Check build status")
def check_build_status_tool(build_id: str) -> str:
    """Return the current status of a build: queued, running, passed, or failed."""
    # Replace this with a call to your own build system.
    return my_ci_client.get_build(build_id).status


agent = Agent(
    role="Build Monitor",
    goal="Start the build and report its final status",
    backstory="An engineer who knows that builds take time.",
    tools=[wait_tool, check_build_status_tool],
)
```

A single call waits at most `max_seconds`. Longer requests are capped and the result says so,
so the agent can simply call the tool again:

```python
wait_tool.run(seconds=3600)
# 'Waited 300 seconds. Requested 3600 seconds, capped at 300 seconds per call -
#  call this tool again if more waiting is needed.'

WaitTool(max_seconds=1800).run(seconds=900)
# 'Waited 900 seconds.'
```

Async execution is supported and does not block the event loop:

```python
import asyncio


async def main():
    print(await wait_tool.arun(seconds=30, reason="waiting for the sandbox build"))


asyncio.run(main())
```
