# ReserpSearchTool

`ReserpSearchTool` exposes the [Reserp Google Search API](https://reserp.ai/) as a minimal CrewAI tool.

It accepts a complete Google Search URL, makes one request, and returns the public response without filtering or reshaping it. The tool deliberately adds no retries, timeout policy, concurrency management, caching, queues, or automatic pagination.

## Authentication

```bash
export RESERP_API_KEY='your-api-key'
```

## Usage

```python
from crewai_tools import ReserpSearchTool

tool = ReserpSearchTool()
result = tool.run(url="https://www.google.com/search?q=photonic+computing&gl=us&hl=en")
print(result)
```

The surrounding application owns retries and all other operational policy. See the [Reserp API documentation](https://reserp.ai/docs) for the complete request, response, error, and pagination contract.
