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
