# Xquik Tools

Use Xquik tools to search and inspect public X/Twitter data from CrewAI agents.

## Tools

- `XquikSearchTweetsTool`: Search public posts with X query operators.
- `XquikGetTweetTool`: Look up one post by tweet ID.
- `XquikGetUserTool`: Look up one profile by username or user ID.
- `XquikGetUserTweetsTool`: List recent posts from one user.
- `XquikGetTrendsTool`: Get regional trending topics by WOEID.

## Setup

Set your Xquik API key:

```bash
export XQUIK_API_KEY="your-api-key"
```

## Example

```python
from crewai_tools import XquikSearchTweetsTool

tool = XquikSearchTweetsTool()
results = tool.run(search_query="from:crewai latest AI agents", limit=10)
```
