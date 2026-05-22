# NotteFunctionTool

Invoke a pre-deployed [Notte Function](https://docs.notte.cc) (serverless workflow) from a CrewAI agent.

## Why this exists alongside `NotteBrowserTool`

`NotteBrowserTool` drives a live browser session with natural-language instructions. That is great for ad-hoc exploration, but every run re-derives the action plan from scratch.

`NotteFunctionTool` invokes a workflow whose code already lives, source-controlled and tested, on the Notte platform. The agent picks *when* to invoke and *what variables to pass*. The function definition itself stays in reviewable, versioned code, not in the LLM's prompt.

Use the browser tool for exploration. Use the function tool for production-grade execution of known tasks.

## Installation

```bash
uv add notte-sdk
# or
pip install notte-sdk
```

## Authentication

Sign up at [notte.cc](https://notte.cc) and export your key:

```bash
export NOTTE_API_KEY="your-notte-api-key"
```

## Usage

```python
from crewai import Agent, Crew, Task
from crewai_tools import NotteFunctionTool

# Bind to a specific deployed function at construction time.
scrape_pricing = NotteFunctionTool(
    function_id="fn_abc123",
    description=(
        "Run the deployed `scrape-pricing` function on Notte. "
        "Takes `vendor` (str) and `tier` (str) as input variables."
    ),
)

analyst = Agent(
    role="Pricing Analyst",
    goal="Compile competitor pricing into a single weekly report",
    backstory="I monitor SaaS pricing across the market.",
    tools=[scrape_pricing],
)

task = Task(
    description=(
        "Run the pricing function for vendor='acme' and tier='pro', "
        "then format the result as a markdown table."
    ),
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task])
print(crew.kickoff())
```

## Configuration

| Parameter | Default | Purpose |
| --- | --- | --- |
| `function_id` | required | The identifier of the deployed function. |
| `api_key` | `os.getenv("NOTTE_API_KEY")` | API key for Notte. |
| `decryption_key` | `None` | Optional decryption key for end-to-end-encrypted functions. |
| `timeout` | `None` | Per-run timeout in seconds. Defaults to the SDK's built-in value. |
| `raise_on_failure` | `False` | Raise on a failed run rather than returning an error result. |

## Output

The tool returns a JSON-encoded result with `success`, `data`, `error`, and `function_run_id` fields. `function_run_id` lets downstream calls retrieve the full run metadata via the Notte API.

## Links

- Notte product: <https://notte.cc>
- Notte Functions documentation: <https://docs.notte.cc>
- Notte Python SDK on PyPI: [`notte-sdk`](https://pypi.org/project/notte-sdk/)
