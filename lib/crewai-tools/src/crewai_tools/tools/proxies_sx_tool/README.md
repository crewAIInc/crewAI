# Proxies.sx Tools

## Description

These tools integrate [Proxies.sx](https://agents.proxies.sx) mobile proxy and stealth browser MCP servers with CrewAI.

- `ProxiesSxProxyTool`: Executes `get_proxy`, `rotate_ip`, and `check_status`.
- `ProxiesSxBrowserTool`: Executes browser actions from `@proxies-sx/browser-mcp`.

Both tools use `npx` to start the MCP server packages on demand.

## Installation

Install CrewAI tools with MCP support:

```bash
pip install "crewai[tools]"
uv add "crewai-tools[mcp]"
```

Ensure Node.js and `npx` are available in your environment.

## Example

```python
from crewai_tools import ProxiesSxBrowserTool, ProxiesSxProxyTool

proxy_tool = ProxiesSxProxyTool(country="US")
browser_tool = ProxiesSxBrowserTool()

proxy_result = proxy_tool.run(action="get_proxy", arguments={})
session = browser_tool.run(action="browser_create", arguments={})
```
