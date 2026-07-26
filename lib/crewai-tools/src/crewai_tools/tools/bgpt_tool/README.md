# BGPT Paper Tool

Search scientific papers with structured full-text experimental evidence via [BGPT](https://bgpt.pro/mcp/).

## REST API (not MCP transport)

- Search: `POST https://bgpt.pro/api/mcp-search`
- DOI lookup: `POST https://bgpt.pro/api/mcp-doi-lookup`

Free tier works without an API key (50 results per network).

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BGPT_API_KEY` | No | Optional paid-tier API key |

## Usage

```python
from crewai_tools import BGPTPaperTool

tool = BGPTPaperTool()

# Keyword search
result = tool.run(search_query="semaglutide cardiovascular outcomes", num_results=3)

# DOI lookup
result = tool.run(doi="10.1038/s41586-024-07386-0")
```

Returns structured evidence fields: methods, sample sizes, limitations, conflicts of interest, falsifiability prompts, and more.
