---
title: crewAI Tools
description: Understanding and leveraging tools within the crewAI framework for agent collaboration and task execution.
---

## Introduction
CrewAI tools empower agents with capabilities ranging from web searching and data analysis to collaboration and delegating tasks among coworkers. This documentation outlines how to create, integrate, and leverage these tools within the CrewAI framework, including a new focus on collaboration tools.

## What is a Tool?
!!! note "Definition"
    A tool in CrewAI is a skill or function that agents can utilize to perform various actions. This includes tools from the [crewAI Toolkit](https://github.com/joaomdmoura/crewai-tools) and [LangChain Tools](https://python.langchain.com/docs/integrations/tools), enabling everything from simple searches to complex interactions and effective teamwork among agents.

## Key Characteristics of Tools

- **Utility**: Designed for various tasks such as web searching, data analysis, content generation, and agent collaboration.
- **Integration**: Enhances agent capabilities by integrating tools directly into their workflow.
- **Customizability**: Offers flexibility to develop custom tools or use existing ones, catering to specific agent needs.

## Using crewAI Tools

crewAI comes with a series to built-in tools that can be used to extend the capabilities of your agents. Start by installing our extra tools package:

```bash
pip install 'crewai[tools]'
```

Here is an example on how to use them:

```python
import os
from crewai import Agent, Task, Crew
# Importing some of the crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SeperDevTool,
    WebsiteSearchTool
)

# get a free account in serper.dev
os.environ["SERPER_API_KEY"] = "Your Key"
os.environ["OPENAI_API_KEY"] = "Your Key"

# Instantiate tools
# Assumes this ./blog-posts exists with existing blog posts on it
docs_tools = DirectoryReadTool(directory='./blog-posts')
file_read_tool = FileReadTool()
search_tool = SeperDevTool()
website_rag = WebsiteSearchTool()

# Create agents
researcher = Agent(
    role='Market Research Analyst',
    goal='Provide up-to-date market analysis of the AI industry',
    backstory='An expert analyst with a keen eye for market trends.',
    tools=[search_tool, website_rag],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Write amazing, super engaging blog post about the AI industry',
    backstory='A skilled writer with a passion for technology.',
    tools=[docs_tools, file_read_tool],
    verbose=True
)

# Create  tasks
research = Task(
    description='Research the AI industry and provide a summary of the latest most trending matters and developments.',
    expected_output='A summary of the top 3 latest most trending matters and developments in the AI industry with you unique take on why they matter.',
    agent=researcher
)

write = Task(
    description='Write an engaging blog post about the AI industry, using the summary provided by the research analyst. Read the latest blog posts in the directory to get inspiration.',
    expected_output='A 4 paragraph blog post formatted as markdown with proper subtitles about the latest trends that is engaging and informative and funny, avoid complex words and make it easy to read.',
    agent=writer,
    output_file='blog-posts/new_post.md' # The final blog post will be written here
)


# Create a crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=2
)

# Execute the tasks
crew.kickoff()
```

## Available crewAI Tools

Most of the tools in the crewAI toolkit offer the ability to set specific arguments or let them to be more wide open, this is the case for most of the tools, for example:

```python
from crewai_tools import DirectoryReadTool

# This will allow the agent with this tool to read any directory it wants during it's execution
tool = DirectoryReadTool()

# OR

# This will allow the agent with this tool to read only the directory specified during it's execution
toos = DirectoryReadTool(directory='./directory')
```

Specific per tool docs are coming soon.
Here is a list of the available tools and their descriptions:

| Tool                        | Description                                                                                   |
| :-------------------------- | :-------------------------------------------------------------------------------------------- |
| **CodeDocsSearchTool**      | A RAG tool optimized for searching through code documentation and related technical documents.|
| **CSVSearchTool**           | A RAG tool designed for searching within CSV files, tailored to handle structured data.       |
| **DirectorySearchTool**     | A RAG tool for searching within directories, useful for navigating through file systems.      |
| **DOCXSearchTool**          | A RAG tool aimed at searching within DOCX documents, ideal for processing Word files.          |
| **DirectoryReadTool**       | Facilitates reading and processing of directory structures and their contents.                |
| **FileReadTool**            | Enables reading and extracting data from files, supporting various file formats.              |
| **GithubSearchTool**        | A RAG tool for searching within GitHub repositories, useful for code and documentation search.|
| **SeperDevTool**            | A specialized tool for development purposes, with specific functionalities under development. |
| **TXTSearchTool**           | A RAG tool focused on searching within text (.txt) files, suitable for unstructured data.      |
| **JSONSearchTool**          | A RAG tool designed for searching within JSON files, catering to structured data handling.     |
| **MDXSearchTool**           | A RAG tool tailored for searching within Markdown (MDX) files, useful for documentation.      |
| **PDFSearchTool**           | A RAG tool aimed at searching within PDF documents, ideal for processing scanned documents.    |
| **PGSearchTool**            | A RAG tool optimized for searching within PostgreSQL databases, suitable for database queries. |
| **RagTool**                 | A general-purpose RAG tool capable of handling various data sources and types.                 |
| **ScrapeElementFromWebsiteTool** | Enables scraping specific elements from websites, useful for targeted data extraction.     |
| **ScrapeWebsiteTool**       | Facilitates scraping entire websites, ideal for comprehensive data collection.                 |
| **WebsiteSearchTool**       | A RAG tool for searching website content, optimized for web data extraction.                   |
| **XMLSearchTool**           | A RAG tool designed for searching within XML files, suitable for structured data formats.      |
| **YoutubeChannelSearchTool**| A RAG tool for searching within YouTube channels, useful for video content analysis.           |
| **YoutubeVideoSearchTool**  | A RAG tool aimed at searching within YouTube videos, ideal for video data extraction.          |

## Creating your own Tools
!!! example "Custom Tool Creation"
    Developers can craft custom tools tailored for their agent’s needs or utilize pre-built options:

To create your own crewAI tools you will need to install our extra tools package:

```bash
pip install 'crewai[tools]'
```

Once you do that there are two main ways for one to create a crewAI tool:

### Subclassing `BaseTool`

```python
from crewai_tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "Clear description for what this tool is useful for, you agent will need this information to use it."

    def _run(self, argument: str) -> str:
        # Implementation goes here
        pass
```

Define a new class inheriting from `BaseTool`, specifying `name`, `description`, and the `_run` method for operational logic.


### Utilizing the `tool` Decorator

For a simpler approach, create a `Tool` object directly with the required attributes and a functional logic.

```python
from crewai_tools import tool
@tool("Name of my tool")
def my_tool(question: str) -> str:
    """Clear description for what this tool is useful for, you agent will need this information to use it."""
    # Function logic here
```

```python
import json
import requests
from crewai import Agent
from crewai.tools import tool
from unstructured.partition.html import partition_html

    # Annotate the function with the tool decorator from crewAI
@tool("Integration with a given API")
def integtation_tool(argument: str) -> str:
    """Integration with a given API"""
    # Code here
    return resutls # string to be sent back to the agent

# Assign the scraping tool to an agent
agent = Agent(
    role='Research Analyst',
    goal='Provide up-to-date market analysis',
    backstory='An expert analyst with a keen eye for market trends.',
    tools=[integtation_tool]
)
```

## Using LangChain Tools
!!! info "LangChain Integration"
    CrewAI seamlessly integrates with LangChain’s comprehensive toolkit for search-based queries and more, here are the available built-in tools that are offered by Langchain [LangChain Toolkit](https://python.langchain.com/docs/integrations/tools/)
 :

```python
from crewai import Agent
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper

# Setup API keys
os.environ["SERPER_API_KEY"] = "Your Key"

search = GoogleSerperAPIWrapper()

# Create and assign the search tool to an agent
serper_tool = Tool(
  name="Intermediate Answer",
  func=search.run,
  description="Useful for search-based queries",
)

agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[serper_tool]
)

# rest of the code ...
```

## Conclusion
Tools are crucial for extending the capabilities of CrewAI agents, allowing them to undertake a diverse array of tasks and collaborate efficiently. When building your AI solutions with CrewAI, consider both custom and existing tools to empower your agents and foster a dynamic AI ecosystem.
