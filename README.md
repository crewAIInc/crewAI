<div align="center">

![Logo of crewAI, two people rowing on a boat](./assets/crewai_logo.png)

<div align="left">

# CrewAI Tools

Empower your CrewAI agents with powerful, customizable tools to elevate their capabilities and tackle sophisticated, real-world tasks.

CrewAI Tools provide the essential functionality to extend your agents, helping you rapidly enhance your automations with reliable, ready-to-use tools or custom-built solutions tailored precisely to your needs.

---

## Quick Links

[Homepage](https://www.crewai.com/) | [Documentation](https://docs.crewai.com/) | [Examples](https://github.com/crewAIInc/crewAI-examples) | [Community](https://community.crewai.com/)

---

## Available Tools

CrewAI provides an extensive collection of powerful tools ready to enhance your agents:

- **File Management**: `FileReadTool`, `FileWriteTool`
- **Web Scraping**: `ScrapeWebsiteTool`, `SeleniumScrapingTool`
- **Database Integrations**: `PGSearchTool`, `MySQLSearchTool`
- **Vector Database Integrations**: `MongoDBVectorSearchTool`, `QdrantVectorSearchTool`, `WeaviateVectorSearchTool`
- **API Integrations**: `SerperApiTool`, `EXASearchTool`
- **AI-powered Tools**: `DallETool`, `VisionTool`, `StagehandTool`

And many more robust tools to simplify your agent integrations.

---

## Creating Custom Tools

CrewAI offers two straightforward approaches to creating custom tools:

### Subclassing `BaseTool`

Define your tool by subclassing:

```python
from crewai.tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "Tool Name"
    description: str = "Detailed description here."

    def _run(self, *args, **kwargs):
        # Your tool logic here
```

### Using the `tool` Decorator

Quickly create lightweight tools using decorators:

```python
from crewai import tool

@tool("Tool Name")
def my_custom_function(input):
    # Tool logic here
    return output
```

---

## CrewAI Tools and MCP

CrewAI Tools supports the Model Context Protocol (MCP). It gives you access to thousands of tools from the hundreds of MCP servers out there built by the community.

Before you start using MCP with CrewAI tools, you need to install the `mcp` extra dependencies:

```bash
pip install crewai-tools[mcp]
# or
uv add crewai-tools --extra mcp
```

To quickly get started with MCP in CrewAI you have 2 options:

### Option 1: Fully managed connection

In this scenario we use a contextmanager (`with` statement) to start and stop the the connection with the MCP server.
This is done in the background and you only get to interact with the CrewAI tools corresponding to the MCP server's tools.

For an STDIO based MCP server:

```python
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

serverparams = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with MCPServerAdapter(serverparams) as tools:
    # tools is now a list of CrewAI Tools matching 1:1 with the MCP server's tools
    agent = Agent(..., tools=tools)
    task = Task(...)
    crew = Crew(..., agents=[agent], tasks=[task])
    crew.kickoff(...)
```
For an SSE based MCP server:

```python
serverparams = {"url": "http://localhost:8000/sse"}
with MCPServerAdapter(serverparams) as tools:
    # tools is now a list of CrewAI Tools matching 1:1 with the MCP server's tools
    agent = Agent(..., tools=tools)
    task = Task(...)
    crew = Crew(..., agents=[agent], tasks=[task])
    crew.kickoff(...)
```

### Option 2: More control over the MCP connection

If you need more control over the MCP connection, you can instanciate the MCPServerAdapter into an `mcp_server_adapter` object which can be used to manage the connection with the MCP server and access the available tools.

**important**: in this case you need to call `mcp_server_adapter.stop()` to make sure the connection is correctly stopped. We recommend that you use a `try ... finally` block run to make sure the `.stop()` is called even in case of errors.

Here is the same example for an STDIO MCP Server:

```python
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

serverparams = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

try:
    mcp_server_adapter = MCPServerAdapter(serverparams)
    tools = mcp_server_adapter.tools
    # tools is now a list of CrewAI Tools matching 1:1 with the MCP server's tools
    agent = Agent(..., tools=tools)
    task = Task(...)
    crew = Crew(..., agents=[agent], tasks=[task])
    crew.kickoff(...)

# ** important ** don't forget to stop the connection
finally: 
    mcp_server_adapter.stop()
```

And finally the same thing but for an SSE MCP Server:

```python
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

serverparams = {"url": "http://localhost:8000/sse"}

try:
    mcp_server_adapter = MCPServerAdapter(serverparams)
    tools = mcp_server_adapter.tools
    # tools is now a list of CrewAI Tools matching 1:1 with the MCP server's tools
    agent = Agent(..., tools=tools)
    task = Task(...)
    crew = Crew(..., agents=[agent], tasks=[task])
    crew.kickoff(...)

# ** important ** don't forget to stop the connection
finally: 
    mcp_server_adapter.stop()
```

### Considerations & Limitations

#### Staying Safe with MCP

Always make sure that you trust the MCP Server before using it. Using an STDIO server will execute code on your machine. Using SSE is still not a silver bullet with many injection possible into your application from a malicious MCP server.

#### Limitations

* At this time we only support tools from MCP Server not other type of primitives like prompts, resources...
* We only return the first text output returned by the MCP Server tool using `.content[0].text`

---

## Why Use CrewAI Tools?

- **Simplicity & Flexibility**: Easy-to-use yet powerful enough for complex workflows.
- **Rapid Integration**: Seamlessly incorporate external services, APIs, and databases.
- **Enterprise Ready**: Built for stability, performance, and consistent results.

---

## Contribution Guidelines

We welcome contributions from the community!

1. Fork and clone the repository.
2. Create a new branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m 'Add my feature'`).
4. Push your branch (`git push origin feature/my-feature`).
5. Open a pull request.

---

## Developer Quickstart

```shell
pip install crewai[tools]
```

### Development Setup

- Install dependencies: `uv sync`
- Run tests: `uv run pytest`
- Run static type checking: `uv run pyright`
- Set up pre-commit hooks: `pre-commit install`

---

## Support and Community

Join our rapidly growing community and receive real-time support:

- [Discourse](https://community.crewai.com/)
- [Open an Issue](https://github.com/crewAIInc/crewAI/issues)

Build smarter, faster, and more powerful AI solutions—powered by CrewAI Tools.
