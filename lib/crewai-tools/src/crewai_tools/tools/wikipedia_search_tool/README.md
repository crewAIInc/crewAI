# WikipediaSearchTool

The **WikipediaSearchTool** is a utility component of the `crewai_tools` package designed to search and retrieve article summaries or full content and metadata from Wikipedia. It allows agents to perform knowledge retrieval across Wikipedia topics with configurable search limits, multi-language support, full article content toggling, and built-in handling for disambiguation and missing pages.

---

## Description

This tool:

* Accepts a **search query** and queries the official Wikipedia API.
* Allows configuration of the **language** (e.g., `'en'`, `'es'`, `'fr'`, `'de'`, `'tr'`).
* Configures the **maximum number of results** to fetch.
* Optionally toggles between lead section **summaries** and **full article content**.
* Sets a compliant **User-Agent** header to adhere to Wikipedia API policies.
* Safely handles **disambiguation pages** and missing articles.
* Returns clean formatted output containing article **titles**, **URLs**, and **summaries** or **full contents**.

---

## Installation

Install the `crewai-tools` package along with the `beautifulsoup4` and `requests` dependencies:

```bash
# Using pip
pip install 'crewai[tools]' beautifulsoup4 requests

# Or using uv
uv add crewai-tools beautifulsoup4 requests
```

---

## Arguments

### Runtime Arguments (`_run`)

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `search_query` | `str` | ✅ | Search query string (e.g., `"Artificial Intelligence"` or `"Quantum Computing"`). |
| `lang` | `str` | ❌ | Optional Wikipedia language code (e.g., `'en'`, `'es'`, `'tr'`). Defaults to tool config. |
| `limit` | `int` | ❌ | Optional maximum results to fetch (between 1 and 10). Defaults to tool config. |
| `load_full_content` | `bool` | ❌ | Optional flag. If `True`, fetches full article text instead of summary. Defaults to `False`. |

### Constructor Arguments

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `lang` | `str` | ❌ | Default Wikipedia language code. Defaults to `"en"`. |
| `limit` | `int` | ❌ | Default number of search results to return. Defaults to `3`. |
| `load_full_content` | `bool` | ❌ | Default setting for loading full article content. Defaults to `False`. |
| `user_agent` | `str` | ❌ | Custom User-Agent header for API requests. Defaults to a standard compliant agent. |

---

## Usage Examples

### 🔧 Basic Initialization

```python
from crewai_tools import WikipediaSearchTool

# Initialize with default settings (English, limit 3, summaries only)
tool = WikipediaSearchTool()

# Perform a search
result = tool.run(search_query="Artificial Intelligence")
print(result)
```

---

### Example 1: Search in a Specific Language

```python
from crewai_tools import WikipediaSearchTool

# Search Wikipedia in Turkish
tool = WikipediaSearchTool(lang="tr")
result = tool.run(search_query="Yapay Zeka")
print(result)
```

---

### Example 2: Fetch Full Article Content

```python
from crewai_tools import WikipediaSearchTool

# Fetch full article text instead of just summaries
tool = WikipediaSearchTool(load_full_content=True, limit=1)
result = tool.run(search_query="Machine Learning")
print(result)
```

---

### Example 3: Integration with a CrewAI Agent

```python
from crewai import Agent, Task, Crew
from crewai_tools import WikipediaSearchTool

wiki_tool = WikipediaSearchTool(lang="en", limit=2)

researcher = Agent(
    role="Academic Researcher",
    goal="Research complex scientific topics using Wikipedia",
    backstory="You are an expert researcher skilled at gathering concise knowledge summaries.",
    tools=[wiki_tool],
    verbose=True
)

task = Task(
    description="Search Wikipedia for 'Quantum Computing' and summarize the key concepts.",
    expected_output="A concise summary of Quantum Computing concepts.",
    agent=researcher
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
print(result)
```

---

## Error Handling

* **Disambiguation Pages**: If a query resolves to a disambiguation page, the tool lists potential matching topics so the agent can refine its query.
* **Missing Pages**: Missing articles are reported gracefully without breaking execution.
* **API Constraints**: Custom User-Agent headers ensure compliance with Wikimedia Foundation API usage guidelines.
