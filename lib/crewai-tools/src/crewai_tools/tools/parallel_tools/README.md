# ParallelSearchTool

A tool that performs web searches using the Parallel Search API. Returns ranked results with compressed excerpts optimized for LLMs.

- **Quickstart**: see the official docs: [Search API Quickstart](https://docs.parallel.ai/search/search-quickstart)
- **Best Practices**: [Search API Best Practices](https://docs.parallel.ai/search/best-practices)

## Installation

To use the `ParallelSearchTool`, you need to install the `parallel-web` library:

```shell
pip install 'crewai[tools]' parallel-web
```

## Why this tool

- **Single-call pipeline**: Replaces search -> scrape -> extract with a single, low-latency API call.
- **LLM-ready**: Returns compressed excerpts that feed directly into LLM prompts (fewer tokens, less pre/post-processing).
- **Flexible**: Control result count and excerpt length; optionally restrict sources via `source_policy`.

## Environment

- `PARALLEL_API_KEY` (required)
- `ANTHROPIC_API_KEY` (optional, for the agent example)

## Initialization Parameters

Configure the tool at initialization time:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | env var | Parallel API key (defaults to `PARALLEL_API_KEY` env var) |
| `mode` | str | None | `"one-shot"` (comprehensive) or `"agentic"` (concise for multi-step workflows) |
| `max_results` | int | 10 | Maximum results to return (1-20) |
| `excerpts` | dict | None | `{"max_chars_per_result": 10000, "max_chars_total": 50000}` |
| `fetch_policy` | dict | None | `{"max_age_seconds": 3600}` for content freshness |
| `source_policy` | dict | None | Domain inclusion/exclusion policy |

## Run Parameters

Pass these when calling `run()`:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `objective` | str | One of these | Natural-language research goal (<= 5000 chars) |
| `search_queries` | list[str] | required | Up to 5 keyword queries (each <= 200 chars) |

Notes:
- At least one of `objective` or `search_queries` is required.
- API is in beta; default rate limit is 600 RPM. Contact support@parallel.ai for production capacity.

## Direct usage

```python
from crewai_tools import ParallelSearchTool

# Initialize with configuration
tool = ParallelSearchTool(
    mode="one-shot",
    max_results=5,
    excerpts={"max_chars_per_result": 10000},
)

# Run a search
resp_json = tool.run(
    objective="When was the United Nations established? Prefer UN's websites.",
    search_queries=["Founding year UN", "Year of founding United Nations"],
)
print(resp_json)  # => {"search_id": ..., "results": [{"url", "title", "excerpts": [...]}, ...]}
```

### Example with `source_policy`

```python
from crewai_tools import ParallelSearchTool

tool = ParallelSearchTool(
    max_results=5,
    excerpts={"max_chars_per_result": 5000},
    source_policy={
        "allow": {"domains": ["un.org"]},
        # "deny": {"domains": ["example.com"]},  # optional
    },
)

resp_json = tool.run(
    objective="When was the United Nations established?",
)
```

### Example with `mode` for agentic workflows

```python
from crewai_tools import ParallelSearchTool

# Use "agentic" mode for multi-step reasoning loops
tool = ParallelSearchTool(
    mode="agentic",  # Concise, token-efficient results
    max_results=5,
)

resp_json = tool.run(
    objective="Find recent research on quantum error correction",
    search_queries=["quantum error correction 2024", "QEC algorithms"],
)
```

### Example with `fetch_policy` for fresh content

```python
from crewai_tools import ParallelSearchTool

# Request content no older than 1 hour
tool = ParallelSearchTool(
    fetch_policy={"max_age_seconds": 3600},
    max_results=10,
)

resp_json = tool.run(
    objective="Latest news on AI regulation",
)
```

## Example with CrewAI Agents

Here's a complete example that uses `ParallelSearchTool` with a CrewAI agent to research a topic and produce a cited answer. 

**Additional requirements:**

```shell
uv add "crewai[anthropic]"
```

**Environment variable:**
- `ANTHROPIC_API_KEY` (for the LLM)

```python
import os
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import ParallelSearchTool

# Configure Claude as the LLM
llm = LLM(
    model="anthropic/claude-opus-4-5-20251101",
    temperature=0.5,
    max_tokens=4096,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Initialize ParallelSearchTool with agentic mode for multi-step workflows
search = ParallelSearchTool(
    mode="agentic",
    max_results=10,
    excerpts={"max_chars_per_result": 10000},
)

# User query
query = "What are current best practices for LLM evaluation and benchmarking?"

# Create a researcher agent
researcher = Agent(
    role="Web Researcher",
    backstory="You are an expert web researcher who finds and synthesizes information from multiple sources.",
    goal="Find high-quality, cited sources and provide accurate, well-sourced answers.",
    tools=[search],
    llm=llm,
    verbose=True,
)

# Define the research task
task = Task(
    description=f"Research: {query}\n\nProvide a comprehensive answer with citations.",
    expected_output="A concise answer with inline citations in the format: [claim] - [source URL]",
    agent=researcher,
    output_file="research_output.md",
)

# Create and run the crew
crew = Crew(
    agents=[researcher],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
)

result = crew.kickoff()
print(result)
```

**Example output:**

```markdown
# Current Best Practices for LLM Evaluation and Benchmarking

## Overview

LLM evaluation is the systematic process of assessing the performance of Large Language
Models to determine their effectiveness, reliability, and efficiency, helping developers
understand the model's strengths and weaknesses while ensuring it functions as expected
in real-world applications - https://www.singlestore.com/blog/complete-guide-to-evaluating-large-language-models/

## Key Evaluation Methodologies

### 1. Multiple-Choice Benchmarks

Multiple-choice benchmarks like MMLU test an LLM's knowledge recall in a straightforward,
quantifiable way similar to standardized tests, measuring accuracy as the fraction of
correctly answered questions - https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches

...
```

**Tips:**
- Use `mode="agentic"` for multi-step reasoning workflows (more concise, token-efficient results)
- Use `mode="one-shot"` for single comprehensive searches
- Increase `excerpts={"max_chars_per_result": N}` for more detailed source content

## Deprecated Parameters

The following parameters are deprecated but still accepted in `run()` for backwards compatibility:

- `processor`: No longer used. Use `mode` at initialization instead.
- `max_chars_per_result`: Use `excerpts={"max_chars_per_result": N}` at initialization instead.

## Behavior

- Single-request web research; no scraping/post-processing required.
- Returns `search_id` and ranked `results` with compressed `excerpts`.
- Uses the `parallel-web` Python SDK with built-in retries, timeouts, and error handling.

## References

- Search API Quickstart: https://docs.parallel.ai/search/search-quickstart
- Best Practices: https://docs.parallel.ai/search/best-practices
