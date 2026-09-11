# Mengram Memory Tools

Give CrewAI agents long-term memory backed by [Mengram](https://mengram.io): semantic (facts), episodic (events), and procedural memory — workflows that learn from successes and failures.

`MengramProceduresTool` is the distinctive one: it returns step-by-step playbooks with a success/failure track record and *preconditions* (assumptions that were violated when the workflow previously failed), so agents stop repeating mistakes they already made.

## Installation

```shell
uv add crewai[tools] mengram-ai
```

Get a free API key at [mengram.io](https://mengram.io) (no card required) and set `MENGRAM_API_KEY`.

## Example

```python
from crewai import Agent
from crewai_tools import MengramSearchTool, MengramSaveTool, MengramProceduresTool

agent = Agent(
    role="DevOps engineer",
    goal="Ship the release safely",
    tools=[
        MengramSearchTool(user_id="user-123"),
        MengramSaveTool(user_id="user-123"),
        MengramProceduresTool(user_id="user-123"),
    ],
)
```

## Arguments

- `api_key` (optional): Mengram API key; defaults to the `MENGRAM_API_KEY` environment variable.
- `user_id` (optional, default `"default"`): scope memories per end-user — one API key, isolated memory per user.
- `limit` (optional, default 5): max results per call.
- `base_url` (optional): self-hosted Mengram instances.
