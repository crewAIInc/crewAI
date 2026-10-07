# Arcmira: YouTube Transcript Search

[API documentation](https://arcmira.com/docs) · [Search reference](https://arcmira.com/docs/search-for-coding-agents) · [OpenAPI](https://api.arcmira.com/v1/openapi.json)

`ArcmiraSearchTool` finds passages in indexed YouTube transcripts. It returns the API response with transcript text, timestamps, source links, partial-result flags and access information so an agent can cite the evidence and describe its limits.

This tool searches Arcmira's index. An empty result does not establish that a topic was never discussed on YouTube.

## Setup

Install `crewai-tools` in your CrewAI project:

```sh
uv add crewai-tools
```

Create an Arcmira API key using the [API quickstart](https://arcmira.com/docs), then set `ARCMIRA_API_KEY` in your environment. Keep credentials outside prompts and source control. You can also pass `api_key` when constructing the tool; it is excluded from serialized tool configuration and model inputs.

Search calls use credits from your plan, then any top-up credits, then your on-demand budget. Check [usage and billing](https://arcmira.com/docs/usage-and-billing) before running the example.

```python
from crewai_tools import ArcmiraSearchTool

tool = ArcmiraSearchTool()
result = tool.run(query="open source", limit=3)
print(result)
```

Pass `tools=[tool]` when constructing your CrewAI agent.

## Inputs

| Input | Description |
| --- | --- |
| `query` | One topic or phrase, at least two characters. Sent as the API's `q` parameter. |
| `limit` | Maximum passages to return, from 1 to 20. Defaults to 5. |

## Results and errors

Preserve `watch_url` and `start_seconds` when citing a passage. Report `partial`, `search_index` and `access` restrictions rather than treating reduced results as complete coverage. The tool returns the API JSON object without summarizing or dropping fields.

Missing credentials fail during setup. HTTP failures raise `RuntimeError` with the status and API error body, including quota, authentication or rate-limit details. Network and non-JSON failures report a short error without raw diagnostics. The tool makes one request per invocation and does not retry internally; an agent may choose to invoke it again.

The tool only calls `GET /v1/search`. It does not read Premium transcripts, create monitors or change webhook settings. For broader research, use [Arcmira's MCP integration](https://arcmira.com/docs/mcp-server).
