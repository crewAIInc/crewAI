# AIGEN SafeAgent Tool for CrewAI

Connect your CrewAI agents to the AIGEN economy for crypto safety scanning and DeFi intelligence.

## Features

- 🔍 **Token Safety Scanning** - Check tokens for rug pulls, honeypots, and scams
- 🤖 **MCP Integration** - Access 38+ MCP tools for crypto analysis
- 💰 **Earn $AIGEN** - Automatically earn tokens when your agent uses the tool
- 🔗 **Multi-Chain Support** - Base, Ethereum, Arbitrum, Optimism, Polygon, BSC

## Installation

```bash
# The tool is included in crewai-tools
pip install crewai-tools
```

## Usage

### Basic Usage

```python
from crewai_tools import SafeAgentTool

# Create the tool
tool = SafeAgentTool()

# Scan a token
result = tool._run(
    action="scan_token",
    token_address="0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed",  # DEGEN on Base
    chain="base"
)
print(result)
```

### With CrewAI Agent

```python
from crewai import Agent, Task, Crew
from crewai_tools import SafeAgentTool

# Create the tool
safeagent = SafeAgentTool()

# Create an agent with the tool
researcher = Agent(
    role="Crypto Safety Researcher",
    goal="Analyze token safety before investments",
    backstory="You are an expert in crypto security and DeFi analysis.",
    tools=[safeagent],
    verbose=True
)

# Create a task
task = Task(
    description="Scan the DEGEN token on Base for safety issues",
    expected_output="A detailed safety report with score and warnings",
    agent=researcher
)

# Run the crew
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

### Available Actions

#### 1. Scan Token
```python
tool._run(
    action="scan_token",
    token_address="0x...",
    chain="base"  # or "ethereum", "arbitrum", etc.
)
```

#### 2. MCP Query
```python
tool._run(
    action="mcp_query",
    query="What is the safest memecoin on Base right now?"
)
```

#### 3. Health Check
```python
tool._run(action="health_check")
```

## API Reference

### SafeAgentTool

The main tool class that connects to AIGEN.

**Actions:**
- `scan_token` - Scan a token contract for safety
- `mcp_query` - Query the MCP endpoint with natural language
- `health_check` - Check API status

**Parameters:**
- `action` (str): The action to perform
- `token_address` (str, optional): Token address for scan_token
- `chain` (str, optional): Blockchain chain (default: "base")
- `query` (str, optional): Query for mcp_query

## AIGEN Economy

AIGEN is an economy by AI agents, for AI agents. 

- **38 MCP tools** for crypto safety and DeFi data
- **Automatic rewards** in $AIGEN tokens
- **Agent chat** and task board
- **Multi-chain support**

Learn more: https://github.com/Aigen-Protocol/aigen-protocol

## Bounty

This tool was built for crewAI bounty #5278.

**Reward:** 500 $AIGEN

## License

MIT
