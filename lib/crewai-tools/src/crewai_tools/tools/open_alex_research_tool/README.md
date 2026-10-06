# OpenAlex Research Tool

The `OpenAlexResearchTool` allows CrewAI agents to search scholarly works using the OpenAlex REST API.

## Features

- Basic searches can be performed without an API key.
- Searches scholarly works across academic disciplines.
- Returns publication title, year, authors, venue, citation count, available open-access URL or DOI, and a short reconstructed abstract.
- Limits results to between 1 and 20 papers.
- Supports an optional email address for OpenAlex's polite pool.

## Usage

```python
from crewai_tools import OpenAlexResearchTool

tool = OpenAlexResearchTool()

result = tool.run(
    query="retrieval augmented generation",
    limit=5,
    email="researcher@example.com",
)

print(result)
