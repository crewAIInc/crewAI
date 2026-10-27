# TavilyExtractorTool

## Description

The `TavilyExtractorTool` allows CrewAI agents to extract structured content from web pages using the Tavily API. It can process single URLs or lists of URLs and provides options for controlling the extraction depth and including images.

## Installation

To use the `TavilyExtractorTool`, you need to install the `tavily-python` library:

```shell
uv add 'crewai[tools]' 'tavily-python>=0.7.24,<0.8'
```

You also need to set your Tavily API key as an environment variable:

```bash
export TAVILY_API_KEY='your-tavily-api-key'
```

## Example

Configure extraction options when constructing the tool. Pass only the URL or URLs to `run`, `arun`, or the agent tool call.

```python
from crewai_tools import TavilyExtractorTool

tool = TavilyExtractorTool(
    extract_depth="advanced",
    query="Pricing and plan limits",
    chunks_per_source=3,
    format="text",
    include_usage=True,
)

result = tool.run(urls=["https://example.com/pricing"])
print(result)
# In async code: await tool.arun(urls=["https://example.com/pricing"])
```

Add the configured tool to an agent's `tools` list to let the agent select URLs. The agent-facing schema remains `urls` only.

## Arguments

Runtime input:

- `urls` (str or list[str], required): One URL or a batch of up to 20 URLs to extract.

Constructor settings:

| Setting | Default | Purpose |
| --- | --- | --- |
| `api_key` | `TAVILY_API_KEY` environment variable | Tavily API key. |
| `proxies` | `None` | Proxy settings for the SDK clients. |
| `extract_depth` | `"basic"` | `"basic"` or `"advanced"` extraction. |
| `include_images` | `False` | Include extracted images. |
| `timeout` | `60` | Request timeout in seconds. |
| `query` | `None` | Rerank extracted content chunks for this intent. |
| `chunks_per_source` | `None` | 1–5 relevant chunks per source; requires `query`. |
| `format` | `None` | `"markdown"` or `"text"`; omitted by default so Tavily uses markdown. |
| `include_favicon` | `None` | Include a favicon URL for each result. |
| `include_usage` | `None` | Include credit usage metadata. |

Optional settings with a `None` value are omitted from requests. Explicit `False` is forwarded in both sync and async calls.

## Response Format

The tool returns the complete SDK response as a JSON string. Check both `results` and `failed_results`: a successful HTTP request can still contain per-URL failures. Result metadata and usage are preserved. SDK errors propagate to the caller.

See the [Extract API reference](https://docs.tavily.com/documentation/api-reference/endpoint/extract) for details on the response structure.
