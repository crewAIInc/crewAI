---
title: crewAI Tools
description: Understanding and leveraging tools within the crewAI framework.
---

## What is a Tool?
!!! note "Definition"
    A tool in CrewAI, is a skill, something Agents can use perform tasks, right now those can be tools from the [crewAI Toolkit](https://github.com/joaomdmoura/crewai-tools) and [LangChain Tools](https://python.langchain.com/docs/integrations/tools), those are basically functions that an agent can utilize for various actions, from simple searches to complex interactions with external systems.

## Key Characteristics of Tools

- **Utility**: Designed for specific tasks such as web searching, data analysis, or content generation.
- **Integration**: Enhance agent capabilities by integrating tools directly into their workflow.
- **Customizability**: Offers the flexibility to develop custom tools or use existing ones from LangChain's ecosystem.

## Creating your own Tools
!!! example "Custom Tool Creation"
    Developers can craft custom tools tailored for their agent’s needs or utilize pre-built options. Here’s how to create one:

```python
import json
import requests
from crewai import Agent
from langchain.tools import tool
from unstructured.partition.html import partition_html

class BrowserTools():

	# Anotate the fuction with the tool decorator from LangChain
	@tool("Scrape website content")
	def scrape_website(website):
		# Write logic for the tool.
		# In this case a function to scrape website content
		url = f"https://chrome.browserless.io/content?token={config('BROWSERLESS_API_KEY')}"
		payload = json.dumps({"url": website})
		headers = {'cache-control': 'no-cache', 'content-type': 'application/json'}
		response = requests.request("POST", url, headers=headers, data=payload)
		elements = partition_html(text=response.text)
		content = "\n\n".join([str(el) for el in elements])
		return content[:5000]

# Assign the scraping tool to an agent
agent = Agent(
	role='Research Analyst',
	goal='Provide up-to-date market analysis',
	backstory='An expert analyst with a keen eye for market trends.',
	tools=[BrowserTools().scrape_website]
)
```

## Using LangChain Tools
!!! info "LangChain Integration"
    CrewAI seamlessly integrates with LangChain’s comprehensive toolkit. Assigning an existing tool to an agent is straightforward:

```python
from crewai import Agent
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper
import os

# Setup API keys
os.environ["OPENAI_API_KEY"] = "Your Key"
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
```

## Conclusion
Tools are crucial for extending the capabilities of CrewAI agents, allowing them to undertake a diverse array of tasks and collaborate efficiently. When building your AI solutions with CrewAI, consider both custom and existing tools to empower your agents and foster a dynamic AI ecosystem.
