# What is a Tool?

A tool in CrewAI is a function or capability that an agent can utilize to perform actions, gather information, or interact with external systems, behind the scenes tools are [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/).
These tools can be as straightforward as a search function or as sophisticated as integrations with other chains or APIs.

## Key Characteristics of Tools

- **Utility**: Tools are designed to serve specific purposes, such as searching the web, analyzing data, or generating content.
- **Integration**: Tools can be integrated into agents to extend their capabilities beyond their basic functions.
- **Customizability**: Developers can create custom tools tailored to the specific needs of their agents or use pre-built LangChain ones available in the ecosystem.

# Creating your own Tools

You can easily create your own tool using [LangChain Tool Custom Tool Creation](https://python.langchain.com/docs/modules/agents/tools/custom_tools).

Example:
```python
import json
import requests

from crewai import Agent
from langchain.tools import tool
from unstructured.partition.html import partition_html

class BrowserTools():
  @tool("Scrape website content")
  def scrape_website(website):
    """Useful to scrape a website content"""
    url = f"https://chrome.browserless.io/content?token={config('BROWSERLESS_API_KEY')}"
    payload = json.dumps({"url": website})
    headers = {
      'cache-control': 'no-cache',
      'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    elements = partition_html(text=response.text)
    content = "\n\n".join([str(el) for el in elements])

    # Return only the first 5k characters
    return content[:5000]


# Create an agent and assign the scrapping tool
agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[BrowserTools().scrape_website]
)
```

# Using Existing Tools

Check [LangChain Integration](https://python.langchain.com/docs/integrations/tools/) for a set of useful existing tools.
To assign a tool to an agent, you'd provide it as part of the agent's properties during initialization.

```python
from crewai import Agent
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper

# Initialize SerpAPI tool with your API key
os.environ["OPENAI_API_KEY"] = "Your Key"
os.environ["SERPER_API_KEY"] = "Your Key"

search = GoogleSerperAPIWrapper()

# Create tool to be used by agent
serper_tool = Tool(
  name="Intermediate Answer",
  func=search.run,
  description="useful for when you need to ask with search",
)

# Create an agent and assign the search tool
agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[serper_tool]
)
```

# Tool Interaction

Tools enhance an agent's ability to perform tasks autonomously or in collaboration with other agents. For instance, an agent might use a search tool to gather information, then pass that data to another agent specialized in analysis.

# Conclusion

Tools are vital components that expand the functionality of agents within the CrewAI framework. They enable agents to perform a wide range of actions and collaborate effectively with one another. As you build with CrewAI, consider the array of tools you can leverage to empower your agents and how they can be interwoven to create a robust AI ecosystem.
