# iGPT Email Tools

This module provides two optional iGPT integrations for email intelligence:

- `IgptEmailAskTool`: ask natural-language questions over a user's email history.
- `IgptEmailSearchTool`: run keyword/semantic search with optional date filters.

## Installation

```bash
uv add crewai-tools --extra igptai
```

## Configuration

Set the following environment variables:

- `IGPT_API_KEY`
- `IGPT_API_USER`

## Example

```python
from crewai_tools import IgptEmailAskTool, IgptEmailSearchTool

ask_tool = IgptEmailAskTool()
search_tool = IgptEmailSearchTool()

answer = ask_tool.run(
    question="What are the unresolved commitments with ACME this week?",
    output_format="json",
)

matches = search_tool.run(
    query="pricing discussion",
    date_from="2026-02-01",
    date_to="2026-02-28",
    max_results=10,
)
```
