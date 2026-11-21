# Asyntai Tools

## Description

[Asyntai](https://asyntai.com) is an AI chatbot for websites. It answers visitors from the website's own pages, documents and knowledge base, and it captures leads. These tools let a CrewAI agent ask that knowledge base, add knowledge to it, and read the leads and conversations it collected.

| Tool | What it does |
| --- | --- |
| `AsyntaiAskWebsiteTool` | Answers a question from the website's knowledge base |
| `AsyntaiListWebsitesTool` | Lists the websites on the account, with their ids |
| `AsyntaiAddTextTool` | Adds a title and a text to the knowledge base |
| `AsyntaiAddUrlTool` | Reads a web page and adds its content to the knowledge base |
| `AsyntaiListKnowledgeTool` | Lists the entries in the knowledge base |
| `AsyntaiGetLeadsTool` | Reads the visitors who left contact details |
| `AsyntaiGetConversationTool` | Reads the messages of one website chat |

## Installation

```shell
pip install 'crewai[tools]'
```

Get your API key in the Asyntai dashboard under Settings > API. The API needs the Starter plan or higher.

```shell
export ASYNTAI_API_KEY="your-api-key"
```

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import AsyntaiAskWebsiteTool, AsyntaiGetLeadsTool

agent = Agent(
    role="Customer success assistant",
    goal="Answer from our website and follow up with new leads",
    backstory="You know our website and our customers.",
    tools=[AsyntaiAskWebsiteTool(), AsyntaiGetLeadsTool()],
)

task = Task(
    description="Read the 5 newest leads and find what our website says about the return policy.",
    expected_output="The leads and the return policy.",
    agent=agent,
)

print(Crew(agents=[agent], tasks=[task]).kickoff())
```

## Arguments

Every tool accepts these constructor arguments:

- `api_key`: Asyntai API key. Defaults to the `ASYNTAI_API_KEY` environment variable.
- `website_id`: Default Asyntai website. Without it, the primary website is used.
- `api_base_url`: Base URL of the Asyntai API. Defaults to `https://asyntai.com/api/v1`.
- `timeout`: Timeout in seconds for each API request. Defaults to 60.

The tools return a readable error text to the agent and do not raise.
