# PlasmateWebsiteTool

## Description

`PlasmateWebsiteTool` fetches a website and returns compact, LLM-ready content using
[Plasmate](https://github.com/plasmate-labs/plasmate) — an open-source Rust browser
engine — instead of raw HTTP + HTML parsing.

Plasmate strips navigation menus, ads, cookie banners, and boilerplate before content
reaches the agent, delivering **10-100x fewer tokens** than raw HTML.  This directly
reduces LLM costs and context noise in web-reading tasks.

No API key is required.  Plasmate runs locally as a subprocess.

**Measured across 45 real sites:**

| Metric | Value |
|---|---|
| Average compression | 17.7x over raw HTML |
| Peak compression | 77x (TechCrunch) |
| RAM per fetch | ~64 MB (vs ~300 MB for Chrome) |

## Installation

```bash
pip install plasmate 'crewai[tools]'
```

## Usage

### Dynamic URL (agent chooses)

```python
from crewai_tools import PlasmateWebsiteTool

tool = PlasmateWebsiteTool()
```

### Fixed URL (scoped to one source)

```python
from crewai_tools import PlasmateWebsiteTool

tool = PlasmateWebsiteTool(
    website_url="https://docs.crewai.com",
    output_format="markdown",
)
```

### In a crew

```python
from crewai import Agent, Task, Crew
from crewai_tools import PlasmateWebsiteTool

researcher = Agent(
    role="Web Researcher",
    goal="Extract information from websites efficiently",
    tools=[PlasmateWebsiteTool()],
    backstory="You retrieve web content with minimal token usage.",
)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `website_url` | `str` | `None` | If set, locks the URL — agent cannot override it |
| `output_format` | `str` | `"markdown"` | `markdown`, `text`, `som` (structured JSON), or `links` |
| `timeout` | `int` | `30` | Per-request timeout in seconds |
| `selector` | `str` | `None` | ARIA role or CSS id to scope extraction (e.g. `"main"`, `"#article"`) |
| `extra_headers` | `dict` | `{}` | HTTP headers forwarded with each request |

## Output formats

| Format | Description | Best for |
|---|---|---|
| `markdown` | Clean markdown with headings and paragraphs | General reading, Q&A |
| `text` | Plain text, no formatting | Summarisation, classification |
| `som` | Full Structured Object Model JSON | Custom extraction, structured data |
| `links` | Extracted hyperlinks only | Link discovery, crawl planning |

## Comparison with ScrapeWebsiteTool

| | ScrapeWebsiteTool | PlasmateWebsiteTool |
|---|---|---|
| Engine | httpx + BeautifulSoup | Plasmate (Rust) |
| Output | Raw text from HTML | Pre-processed markdown / text / SOM |
| Tokens per page (avg) | ~75,000 | ~4,200 |
| API key required | No | No |
| JS rendering | No | No |

For JavaScript-heavy SPAs, use `ScrapeWebsiteTool` or `BrowserbaseLoadTool`.
