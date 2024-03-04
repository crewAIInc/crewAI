---
title: Creating your own Tools
description: Guide on how to create and use custom tools within the crewAI framework.
---

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
        return "Result from custom tool"
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

### Using the 'Tool' function from langchain

For another simple approach, create a function in python directly with the required attributes and a functional logic.

```python
def combine(a, b):
    return a + b
```

Then you can add that function into the your tool by using 'func' variable in the Tool function.

```python
from langchain.agents import Tool

math_tool = Tool(
            name="Math tool",
            func=math_tool,
            description="Useful for adding two numbers together, in other words combining them."
        )
```
