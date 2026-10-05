# Hacker News Top Stories Tool

The `HackerNewsTopStoriesTool` allows CrewAI agents to retrieve top stories from Hacker News using the official public Firebase REST API.

## Description

This tool fetches current top story titles, upvote scores, and URLs without requiring authentication or an external API key.

The tool accepts a `limit` between 1 and 20 and skips invalid, deleted, dead, null, and non-story items.

## Usage

```python
from crewai_tools import HackerNewsTopStoriesTool

tool = HackerNewsTopStoriesTool()

# Fetch top 5 stories (default limit)
result = tool.run(limit=5)
print(result)
```

The `limit` parameter must be between 1 and 20.

