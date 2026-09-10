# iFlow Search Tools

Tools for the [iFlow Search (心流搜索)](https://platform.iflow.cn/) API, following the
endpoint-specific pattern used by the Brave Search suite:

| Tool | Capability |
|---|---|
| `IFlowWebSearchTool` | Web search |
| `IFlowImageSearchTool` | Image search |
| `IFlowWebFetchTool` | Web page fetch / content extraction |

## Requirements

- `pip install crewai-tools[iflow-search]` — installs both the tool classes
  and the underlying `iflow-search` SDK. (If you already have `crewai-tools`,
  just add the SDK with `pip install iflow-search` or `uv add iflow-search`.)
- `IFLOW_API_KEY` environment variable (or pass `api_key=` to the tool)

## Usage

```python
from crewai_tools import IFlowWebSearchTool, IFlowImageSearchTool, IFlowWebFetchTool

web_search = IFlowWebSearchTool()
print(web_search.run(query="crewai multi-agent framework", count=5))

image_search = IFlowImageSearchTool()
print(image_search.run(query="agent orchestration diagrams"))

web_fetch = IFlowWebFetchTool()
print(web_fetch.run(url="https://docs.crewai.com"))
```

All tools return structured JSON strings. Construction-time options:
`api_key`, `base_url`, `timeout`, or a pre-built `client`
(`iflow_search.IFlowSearchClient`).
