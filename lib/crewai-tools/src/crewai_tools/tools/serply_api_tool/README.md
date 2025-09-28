# Serply API Documentation

## Description
This tool is designed to perform a web/news/scholar search for a specified query from a text's content across the internet. It utilizes the [Serply.io](https://serply.io) API to fetch and display the most relevant search results based on the query provided by the user.

## Installation

To incorporate this tool into your project, follow the installation instructions below:
```shell
pip install 'crewai[tools]'
```

## Examples

## Web Search
The following example demonstrates how to initialize the tool and execute a search the web with a given query:

```python
from crewai_tools import SerplyWebSearchTool

# Initialize the tool for internet searching capabilities
tool = SerplyWebSearchTool()

# increase search limits to 100 results
tool = SerplyWebSearchTool(limit=100)


# change results language (fr - French)
tool = SerplyWebSearchTool(hl="fr")
```

## News Search
The following example demonstrates how to initialize the tool and execute a search news with a given query:

```python
from crewai_tools import SerplyNewsSearchTool

# Initialize the tool for internet searching capabilities
tool = SerplyNewsSearchTool()

# change country news (JP - Japan)
tool = SerplyNewsSearchTool(proxy_location="JP")
```

## Scholar Search
The following example demonstrates how to initialize the tool and execute a search scholar articles a given query:

```python
from crewai_tools import SerplyScholarSearchTool

# Initialize the tool for internet searching capabilities
tool = SerplyScholarSearchTool()

# change country news (GB - Great Britain)
tool = SerplyScholarSearchTool(proxy_location="GB")
```

## Job Search
The following example demonstrates how to initialize the tool and searching for jobs in the USA:

```python
from crewai_tools import SerplyJobSearchTool

# Initialize the tool for internet searching capabilities
tool = SerplyJobSearchTool()
```


## Web Page To Markdown
The following example demonstrates how to initialize the tool and fetch a web page and convert it to markdown:

```python
from crewai_tools import SerplyWebpageToMarkdownTool

# Initialize the tool for internet searching capabilities
tool = SerplyWebpageToMarkdownTool()

# change country make request from (DE - Germany)
tool = SerplyWebpageToMarkdownTool(proxy_location="DE")
```

## Combining Multiple Tools

The following example demonstrates performing a Google search to find relevant articles. Then, convert those articles to markdown format for easier extraction of key points.

```python
from crewai import Agent
from crewai_tools import SerplyWebSearchTool, SerplyWebpageToMarkdownTool

search_tool = SerplyWebSearchTool()
convert_to_markdown = SerplyWebpageToMarkdownTool()

# Creating a senior researcher agent with memory and verbose mode
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "Driven by curiosity, you're at the forefront of"
    "innovation, eager to explore and share knowledge that could change"
    "the world."
  ),
  tools=[search_tool, convert_to_markdown],
  allow_delegation=True
)
```

## Steps to Get Started
To effectively use the `SerplyApiTool`, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **API Key Acquisition**: Acquire a `serper.dev` API key by registering for a free account at [Serply.io](https://serply.io).
3. **Environment Configuration**: Store your obtained API key in an environment variable named `SERPLY_API_KEY` to facilitate its use by the tool.

## Conclusion
By integrating the `SerplyApiTool` into Python projects, users gain the ability to conduct real-time searches, relevant news across the internet directly from their applications. By adhering to the setup and usage guidelines provided, incorporating this tool into projects is streamlined and straightforward.
