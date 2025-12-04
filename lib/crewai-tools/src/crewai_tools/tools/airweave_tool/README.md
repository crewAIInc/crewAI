# Airweave Search Tools

Search across all your connected data sources (Stripe, GitHub, Notion, Slack, and 50+ more) using Airweave's unified search API.

## Installation

```bash
pip install 'crewai-tools[airweave]'
```

Or install the SDK directly:

```bash
pip install airweave-sdk
```

## Setup

### 1. Get your API key

Sign up at [https://app.airweave.ai](https://app.airweave.ai) and get your API key.

### 2. Set environment variable

```bash
export AIRWEAVE_API_KEY="your_api_key_here"
```

### 3. Create a collection and connect data sources

Through the Airweave dashboard:
1. Create a new collection
2. Add source connections (Stripe, GitHub, Notion, etc.)
3. Wait for initial sync to complete
4. Copy your collection's `readable_id`

## Tools

### AirweaveSearchTool

Basic search tool for straightforward queries. Mirrors the `client.collections.search()` method from the Airweave Python SDK.

**When to use:**
- Simple searches without filtering
- Quick lookups across all data sources
- When you don't need advanced features

**Example:**

```python
from crewai import Agent, Task, Crew
from crewai_tools import AirweaveSearchTool

# Initialize the tool
search_tool = AirweaveSearchTool(
    collection_id="my-collection-id"
)

# Create an agent with the tool
agent = Agent(
    role="Data Analyst",
    goal="Find information from connected data sources",
    tools=[search_tool],
    verbose=True
)

# Create a task
task = Task(
    description="Find all failed payments from the last month",
    agent=agent,
    expected_output="List of failed payments with customer details"
)

# Run the crew
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

**Get AI-generated answers:**

```python
from crewai import Task

task = Task(
    description="""
    What are the most common customer complaints this month?
    Use response_type='completion' to get an AI-generated summary.
    """,
    agent=agent
)
```

### AirweaveAdvancedSearchTool

Advanced search tool with filtering, reranking, and fine-tuned control. Mirrors the `client.collections.search_advanced()` method from the Airweave Python SDK.

**When to use:**
- Filter by specific data sources
- Prioritize recent results
- Set minimum relevance scores
- Enable AI reranking for better results
- Choose specific search methods (hybrid/neural/keyword)

**Example:**

```python
from crewai_tools import AirweaveAdvancedSearchTool

# Initialize with advanced options
advanced_tool = AirweaveAdvancedSearchTool(
    collection_id="my-collection-id"
)

agent = Agent(
    role="Customer Support Analyst",
    goal="Find recent customer issues from specific sources",
    tools=[advanced_tool]
)

task = Task(
    description="""
    Find customer complaints about billing from Zendesk in the last week.
    Use these parameters:
    - source_filter: 'Zendesk'
    - recency_bias: 0.8
    - enable_reranking: True
    - score_threshold: 0.7
    """,
    agent=agent
)
```

## Parameters

### AirweaveSearchTool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | Required | Search query to find relevant information |
| `limit` | int | 10 | Maximum number of results (1-100) |
| `response_type` | str | "raw" | "raw" for search results, "completion" for AI answer |
| `recency_bias` | float | 0.0 | Weight for recent results (0.0-1.0) |

### AirweaveAdvancedSearchTool Parameters

All basic parameters plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_filter` | str | None | Filter by specific source (e.g., "Stripe", "GitHub") |
| `score_threshold` | float | None | Minimum similarity score (0.0-1.0) |
| `recency_bias` | float | 0.3 | Weight for recent results (0.0-1.0) |
| `enable_reranking` | bool | True | Enable AI reranking for better relevance |
| `search_method` | str | "hybrid" | "hybrid", "neural", or "keyword" |

### Tool Configuration (Constructor)

Both tools accept these configuration parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_id` | str | Required | Your collection's readable ID |
| `base_url` | str | None | Custom API URL for self-hosted instances |
| `max_content_length` | int | 300 | Max characters to show per result |

## What Can Airweave Search?

Airweave connects to 50+ data sources including:

### Finance & Billing
- Stripe
- QuickBooks
- Chargebee

### Development Tools
- GitHub
- GitLab
- Jira
- Linear

### Collaboration
- Slack
- Microsoft Teams
- Discord

### Productivity
- Notion
- Google Drive
- Confluence
- Dropbox

### CRM & Support
- Salesforce
- HubSpot
- Zendesk
- Intercom

### And many more...

[See full list of integrations →](https://docs.airweave.ai/integrations)

## Features

✅ **Unified Search** - Search across all data sources with one query  
✅ **Semantic Search** - Natural language understanding  
✅ **Hybrid Search** - Combines vector and keyword search  
✅ **AI Reranking** - Improves result relevance (Advanced)  
✅ **AI Answers** - Get generated answers via `response_type="completion"`  
✅ **Recency Bias** - Prioritize recent results  
✅ **Source Filtering** - Search specific sources only (Advanced)  
✅ **Score Threshold** - Filter by relevance (Advanced)  
✅ **Incremental Sync** - Data stays automatically updated  
✅ **Multi-tenant** - Each user has their own collections  

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIRWEAVE_API_KEY` | Yes | - | Your Airweave API key |

## Response Types

### Raw Results (`response_type="raw"`)

Returns structured search results with:
- Content snippets
- Similarity scores
- Source information
- Entity IDs
- Creation timestamps
- URLs (when available)

### AI Completion (`response_type="completion"`)

Returns an AI-generated natural language answer based on the retrieved documents. Use this when you want:
- Summarized information
- Direct answers to questions
- Synthesized insights from multiple sources

## Search Methods (Advanced Tool)

### Hybrid (Default)
Combines semantic (neural) and keyword (BM25) search for best results.

### Neural
Pure semantic search using embeddings. Best for conceptual queries.

### Keyword
Traditional keyword search. Best for exact term matching.

## Use Cases

### Customer Support Agent
```python
agent = Agent(
    role="Customer Support Specialist",
    goal="Find and resolve customer issues",
    tools=[AirweaveAdvancedSearchTool(collection_id="support-data")],
    backstory="Expert at finding relevant customer data across Zendesk, Slack, and email"
)
```

### Sales Intelligence Agent
```python
agent = Agent(
    role="Sales Intelligence Analyst",
    goal="Research accounts and opportunities",
    tools=[AirweaveSearchTool(collection_id="sales-data")],
    backstory="Analyzes data from Salesforce, HubSpot, and LinkedIn"
)
```

### Financial Analysis Agent
```python
agent = Agent(
    role="Financial Analyst",
    goal="Analyze payment trends and issues",
    tools=[AirweaveAdvancedSearchTool(collection_id="finance-data")],
    backstory="Tracks payments, invoices, and transactions from Stripe and QuickBooks"
)
```

### Technical Documentation Agent
```python
agent = Agent(
    role="Documentation Specialist",
    goal="Answer technical questions from internal docs",
    tools=[AirweaveSearchTool(collection_id="docs-collection")],
    backstory="Searches through Notion, Confluence, and GitHub repos"
)
```

## Error Handling

The tools handle errors gracefully and return clear error messages:

- **Missing API key** - Clear instruction on how to set it
- **Collection not found** - Verifies collection exists
- **No results** - Suggests rephrasing query
- **API errors** - Returns error details for debugging

## Advanced Filtering Examples

### Filter by Multiple Sources (Future Enhancement)

Currently supports single source filtering. For multiple sources, create filter using SDK types:

```python
from airweave import Filter, FieldCondition, MatchAny

# In your agent's task description, specify the filter logic
# The tool currently supports single source_filter parameter
```

### Combine Multiple Filters

For complex filtering needs beyond single source, consider using the Airweave SDK directly within a custom tool or use multiple search calls.

## Best Practices

1. **Start with Basic Search** - Use `AirweaveSearchTool` for most queries
2. **Use Advanced When Needed** - Switch to `AirweaveAdvancedSearchTool` when you need filtering or reranking
3. **Set Appropriate Limits** - Higher limits for comprehensive searches, lower for quick lookups
4. **Enable Reranking for Complex Queries** - Better results for nuanced questions
5. **Use Recency Bias for Time-Sensitive Data** - Great for support tickets, recent transactions
6. **Choose Response Type Based on Need** - "raw" for structured data, "completion" for answers
7. **Filter by Source for Focused Searches** - Reduces noise when you know the source

## Troubleshooting

### "No results found"
- Check that your collection has data synced
- Verify sync jobs completed successfully
- Try broader search terms
- Lower score_threshold if using advanced search

### "Unable to generate an answer"
- Ensure you have relevant data in your collection
- Try response_type="raw" to see what results are available
- Rephrase your query to be more specific

### Import errors
```bash
pip install --upgrade airweave-sdk crewai-tools
```

## Learn More

- [Airweave Documentation](https://docs.airweave.ai)
- [API Reference](https://docs.airweave.ai/api-reference)
- [Python SDK](https://github.com/airweave-ai/python-sdk)
- [Get API Key](https://app.airweave.ai)
- [CrewAI Documentation](https://docs.crewai.com)

## Support

- Discord: [Join Airweave Community](https://discord.gg/airweave)
- Email: support@airweave.ai
- GitHub Issues: [airweave-ai/python-sdk](https://github.com/airweave-ai/python-sdk/issues)


