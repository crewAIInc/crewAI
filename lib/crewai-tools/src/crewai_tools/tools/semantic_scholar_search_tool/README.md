# Semantic Scholar Search Tool

A CrewAI tool for searching academic papers on Semantic Scholar. Supports keyword search, paper lookup by ID, and various filtering options.

## What is Semantic Scholar?

[Semantic Scholar](https://www.semanticscholar.org/) is a free, AI-powered research tool for scientific literature, developed by the Allen Institute for AI (AI2). It provides access to over 214 million papers across all fields of science.

Key capabilities:
- **Academic Search**: Find papers across multiple disciplines
- **Citation Analysis**: Track paper citations and influence metrics
- **Paper Recommendations**: Get relevant paper suggestions based on your research interests
- **Open Access**: Many papers include free PDF links

This tool integrates with the Semantic Scholar Academic Graph API to enable programmatic access to their extensive database of scholarly literature.

## Why Semantic Scholar?

Semantic Scholar stands out among academic search engines for several reasons:

### vs Google Scholar

| Aspect | Google Scholar | Semantic Scholar |
|--------|---------------|------------------|
| **Speed** | Slower, limited API | Fast, dedicated API |
| **AI Features** | Basic | AI-powered relevance ranking |
| **Citation Metrics** | Limited | Detailed citation counts & influence |
| **API Access** | No official API | Full REST API available |
| **Filtering** | Basic | Advanced filters (year, citations, fields) |
| **Open Data** | No | Academic Graph data accessible |

### vs Web of Science / Scopus

| Aspect | Web of Science | Scopus | Semantic Scholar |
|--------|---------------|--------|------------------|
| **Cost** | Expensive subscription | Expensive subscription | **Free** |
| **Coverage** | Selected journals | Selected journals | **All sources** |
| **API Access** | Limited, costly | Limited | **Full API** |
| **AI Enhancement** | No | No | **Yes** |
| **Citation Network** | Yes | Yes | **Yes, with AI insights** |

### Key Advantages

1. **Free Access**: No subscription required for basic use
2. **Comprehensive Coverage**: Indexes papers from all sources, not just selected journals
3. **Programmatic Access**: Full API for integration with applications and agents
4. **AI-Powered Search**: Semantic Scholar's ranker is trained specifically for academic relevance
5. **Rich Metadata**: Returns abstracts, authors, citations, influential citations, fields of study
6. **Modern API**: Well-documented REST API with batch endpoints for efficient querying

For AI agents and automation workflows, Semantic Scholar's API-first approach makes it the ideal choice for academic research tasks.

## Features

- **Keyword Search**: Search academic papers using Semantic Scholar's bulk search endpoint
- **Paper Lookup**: Get detailed information about a specific paper by its Semantic Scholar ID
- **Filtering**: Filter by year range, citation count, fields of study, publication type, open access
- **Sorting**: Sort results by citation count, publication date, or paper ID
- **Flexible Authentication**: Supports optional API key with automatic fallback on invalid keys

## Installation

```bash
pip install crewai-tools
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
from crewai_tools import SemanticScholarSearchTool

# Initialize the tool
tool = SemanticScholarSearchTool()

# Search for papers
result = tool._run(search_query="machine learning", limit=5)
print(result)
```

## API Key Configuration

The tool supports an optional API key for higher rate limits. You can provide it in multiple ways:

### Option 1: Environment Variable

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-api-key"
```

### Option 2: Constructor Parameter

```python
tool = SemanticScholarSearchTool(api_key="your-api-key")
```

### Option 3: Runtime Parameter

```python
result = tool._run(search_query="AI", api_key="your-api-key")
```

If an invalid API key is provided, the tool will automatically fallback to unauthenticated requests.

## Usage Examples

### Basic Search

```python
tool = SemanticScholarSearchTool()
result = tool._run(search_query="generative AI")
print(result)
```

### Search with Year Filter

```python
# Papers from 2023 onwards
result = tool._run(search_query="neural networks", year="2023-")

# Papers from 2020 to 2023
result = tool._run(search_query="deep learning", year="2020-2023")
```

### Search with Sorting

```python
# Sort by citation count (most cited first)
result = tool._run(search_query="AI", sort="citationCount")

# Sort by publication date (newest first)
result = tool._run(search_query="machine learning", sort="publicationDate")
```

### Search with Minimum Citations

```python
# Only papers with at least 100 citations
result = tool._run(search_query="transformers", min_citation_count=100)
```

### Search by Field of Study

```python
# Filter by specific fields
result = tool._run(search_query="neural networks", fields_of_study="Computer Science")
result = tool._run(search_query="quantum", fields_of_study="Physics,Computer Science")
```

### Search by Publication Type

```python
# Filter by publication type
result = tool._run(search_query="AI", publication_types="JournalArticle")
result = tool._run(search_query="machine learning", publication_types="ConferencePaper")
```

### Search for Open Access Papers Only

```python
# Only papers with available PDFs
result = tool._run(search_query="deep learning", open_access_pdf=True)
```

### Custom Fields

```python
# Specify which fields to return
result = tool._run(
    search_query="AI",
    fields="title,url,year,citationCount,authors"
)
```

### Paper Lookup by ID

```python
# Get details of a specific paper
result = tool._run(
    mode="lookup",
    paper_id="649def34f8be52c8b66281af98ae884c09aef38b"
)
print(result)
```

### Combined Options

```python
result = tool._run(
    search_query="large language models",
    year="2022-",
    limit=10,
    sort="citationCount",
    min_citation_count=50,
    fields_of_study="Computer Science",
    publication_types="JournalArticle",
    open_access_pdf=True
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_query` | `str` | Required* | Search query for academic papers |
| `mode` | `str` | `"search"` | Operation mode: `"search"` or `"lookup"` |
| `paper_id` | `str` | None | Semantic Scholar paper ID (required for lookup mode) |
| `year` | `str` | None | Year filter (e.g., `"2023-"`, `"2020-2023"`) |
| `fields` | `str` | `"title,url,year,citationCount,abstract,authors"` | Comma-separated list of fields to return |
| `limit` | `int` | `10` | Maximum number of results (max 1000) |
| `sort` | `str` | None | Sort by: `"citationCount"`, `"publicationDate"`, `"paperId"` |
| `min_citation_count` | `int` | None | Minimum citation count filter |
| `fields_of_study` | `str` | None | Filter by fields of study (e.g., `"Computer Science,Physics"`) |
| `publication_types` | `str` | None | Filter by publication types (e.g., `"JournalArticle,ConferencePaper"`) |
| `open_access_pdf` | `bool` | None | Filter to only papers with open access PDFs |
| `api_key` | `str` | None | Semantic Scholar API key (optional) |

*Either `search_query` (for search mode) or `paper_id` (for lookup mode) is required.

## Available Fields

When requesting paper data, you can specify any of these fields:
- `paperId` - Paper's unique identifier
- `title` - Paper title
- `url` - Paper URL on Semantic Scholar
- `year` - Publication year
- `citationCount` - Number of citations
- `influentialCitationCount` - Number of influential citations
- `abstract` - Paper abstract
- `authors` - List of authors
- `venue` - Publication venue
- `publicationTypes` - Type of publication
- `openAccessPdf` - Open access PDF URL
- `fieldsOfStudy` - Fields of study
- `referenceCount` - Number of references
- `referencePapers` - Reference papers

## Using with CrewAI Agents

```python
from crewai import Agent, Task, Crew
from crewai_tools import SemanticScholarSearchTool

# Create the tool
search_tool = SemanticScholarSearchTool()

# Create an agent
researcher = Agent(
    role="Researcher",
    goal="Find relevant academic papers on a topic",
    tools=[search_tool]
)

# Create a task
task = Task(
    description="Find 5 recent papers on transformer architectures",
    agent=researcher
)

# Run the crew
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## Error Handling

The tool raises exceptions for common HTTP errors:

- `400 Bad Request` - Invalid query parameters
- `401 Unauthorized` - Missing or invalid API key
- `403 Forbidden` - API key rejected (falls back to unauthenticated)
- `404 Not Found` - Paper not found (lookup mode)
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Semantic Scholar server error

## Rate Limits

Without an API key:
- Shared rate limit across all unauthenticated users

With an API key:
- 1 request per second on all endpoints
- Higher limits may be available upon request

## License

MIT License

## Credits

- [Semantic Scholar](https://www.semanticscholar.org/) - Academic search engine
- [CrewAI](https://crewai.com/) - AI agents framework