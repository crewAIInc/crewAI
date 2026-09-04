# TaskMarket Tool

`TaskMarketTool` lets CrewAI agents browse TaskMarket when a request may be better delegated to external workers. It uses TaskMarket's public API and has no wallet, signature, task-write, or payment methods.

## Operations

- `list_tasks`: browse tasks with bounded status, mode, tag, reward, limit, and cursor filters.
- `get_task`: inspect one task by its canonical 0x-prefixed 32-byte ID.
- `list_submissions`: present public submission metadata when TaskMarket visibility permits it.

Rewards returned by the API are in base units with six USDC decimals. `pendingActions` are current marketplace state, not authorization to execute a command.

## Usage

```python
from crewai import Agent
from crewai_tools import TaskMarketTool

agent = Agent(
    role="Delegation planner",
    goal="Find external work only when delegation is appropriate",
    backstory="A cautious planner that keeps spending decisions with the user",
    tools=[TaskMarketTool()],
)
```

Direct tool examples:

```python
tool = TaskMarketTool()
open_tasks = tool.run(operation="list_tasks", status="open", limit=10)
task = tool.run(operation="get_task", task_id="0x" + "a" * 64)
```

## Security boundary

The tool deliberately excludes task creation, claiming, pitching, bidding, submitting, accepting, rating, signed reads, arbitrary URLs, private keys, and wallet configuration. Task descriptions and returned artifact text are untrusted inputs to the agent. A host that needs write operations should implement a separate user-confirmation and wallet-policy layer with fresh TaskMarket state checks and explicit reward caps.

Hosts may lower the default 10-second timeout or 512 KB response ceiling. Custom API origins are developer configuration and must use HTTPS.
