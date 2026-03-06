# YouResearchTool

Performs comprehensive, multi-step research using the You.com Research API.

## Installation

1. Get your API key from https://you.com/platform/api-keys

2. Install dependencies and set your API key:

```bash
uv add requests
export YOU_API_KEY="your-api-key-here"
```

## Usage

```python
from crewai_tools import YouResearchTool

# Basic usage
research_tool = YouResearchTool()
result = research_tool.run(input="What are the latest advances in quantum computing?")

# With effort level control
result = research_tool.run(
    input="What measurable actions improved air quality in major cities over the past decade?",
    research_effort="deep"
)

# Configure default effort level at initialization
research_tool = YouResearchTool(research_effort="exhaustive")
result = research_tool.run(input="Explain the economic impact of renewable energy adoption globally")
```

## Parameters

- **input** (required): The research question or complex query (max 40,000 characters)
- **research_effort**: Controls depth vs. speed tradeoff (default: "standard")
  - `lite`: Fast answers, good for straightforward questions
  - `standard`: Balanced speed and depth, fits most questions
  - `deep`: More thorough cross-referencing, use when accuracy matters
  - `exhaustive`: Most comprehensive, best for complex research tasks
- **timeout**: API request timeout in seconds (default: 120)

## Response Format

```json
{
  "output": {
    "content": "Comprehensive answer with inline citations [[1]], [[2]]...",
    "content_type": "text",
    "sources": [
      {
        "url": "https://example.com/article",
        "title": "Article Title",
        "snippets": ["Relevant excerpt from the source..."]
      }
    ]
  }
}
```

## With CrewAI

```python
from crewai import Agent, Task
from crewai_tools import YouResearchTool

research_tool = YouResearchTool(research_effort="deep")

researcher = Agent(
    role="Research Analyst",
    goal="Conduct thorough research on complex topics",
    backstory="Expert at synthesizing information from multiple sources",
    tools=[research_tool]
)

task = Task(
    description="Research the long-term economic effects of remote work adoption",
    agent=researcher,
    expected_output="Comprehensive analysis with cited sources"
)
```

## Combining with Other You.com Tools

```python
from crewai_tools import YouSearchTool, YouResearchTool, YouContentsTool

# Quick search for recent results
search_tool = YouSearchTool(count=5)

# Deep research for complex questions
research_tool = YouResearchTool(research_effort="deep")

# Extract full content from specific pages
contents_tool = YouContentsTool()
```
