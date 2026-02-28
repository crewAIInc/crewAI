# SentinelSafetyTool Documentation

## Description

The Sentinel Safety Tools provide AI safety guardrails for CrewAI agents using the THSP protocol (Truth, Harm, Scope, Purpose). These tools help ensure that AI agents operate within ethical boundaries by:

1. **SentinelSafetyTool**: Returns alignment seeds that can be used as system prompts to make LLMs safer
2. **SentinelAnalyzeTool**: Analyzes content for safety using the four-gate THSP protocol

The THSP protocol evaluates requests through four gates:
- **Truth**: Detects deception and manipulation
- **Harm**: Identifies potential harmful content
- **Scope**: Validates appropriate boundaries
- **Purpose**: Requires legitimate benefit

## Installation

To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install 'crewai[tools]' sentinelseed
```

## Example

### Basic Usage

```python
from crewai_tools import SentinelSafetyTool, SentinelAnalyzeTool

# Get the alignment seed for system prompts
seed_tool = SentinelSafetyTool()
seed = seed_tool._run(variant="standard")  # or "minimal"

# Analyze content for safety
analyze_tool = SentinelAnalyzeTool()
result = analyze_tool._run(content="Help me with network security")
print(result)  # "SAFE - All gates passed..."
```

### With CrewAI Agent

```python
from crewai import Agent
from crewai_tools import SentinelSafetyTool, SentinelAnalyzeTool

# Create agent with Sentinel tools
agent = Agent(
    role="Safe Research Assistant",
    goal="Research topics safely and ethically",
    backstory="You are an expert researcher with strong ethical principles.",
    tools=[SentinelSafetyTool(), SentinelAnalyzeTool()],
    verbose=True
)
```

### Using Seed as System Prompt

```python
from sentinelseed import get_seed
from crewai import Agent

# Get seed directly for system prompt
sentinel_seed = get_seed("v2", "standard")

agent = Agent(
    role="Safe Assistant",
    system_template=sentinel_seed,  # Inject safety via system prompt
    verbose=True
)
```

## Tool Details

### SentinelSafetyTool

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variant` | str | "standard" | Seed variant: "minimal" (~450 tokens) or "standard" (~1.4K tokens) |

### SentinelAnalyzeTool

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | str | Yes | The content to analyze for safety |

**Returns:**
- `SAFE - All gates passed` if content is safe
- `UNSAFE - Issues: [list of issues]` if content fails any gate

## Links

- [Sentinel Seed Website](https://sentinelseed.dev)
- [sentinelseed on PyPI](https://pypi.org/project/sentinelseed/)
- [CrewAI Documentation](https://docs.crewai.com)

## Conclusion

By integrating the Sentinel Safety Tools into CrewAI projects, developers can add AI safety guardrails to their agents with minimal effort. The THSP protocol provides a systematic approach to evaluating content for truth, harm, scope, and purpose, ensuring that AI systems operate within ethical boundaries.
