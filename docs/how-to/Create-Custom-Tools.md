---
title: Creating your own Tools
description: Guide on how to create and use custom tools within the crewAI framework.
---

## Creating your own Tools
!!! example "Custom Tool Creation"
    Developers can craft custom tools tailored to their agent’s needs or utilize pre-built options.

To create your own crewAI tools, you will need to install our extra tools package:

```bash
pip install 'crewai[tools]'
```

Once installed, there are two primary methods for creating a crewAI tool:

### Subclassing `BaseTool`

To define a custom tool, create a new class that inherits from `BaseTool`. Specify the `name`, `description`, and implement the `_run` method to outline its operational logic.

```python
from crewai_tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "Clear description for what this tool is useful for. Your agent will need this information to utilize it effectively."

    def _run(self, argument: str) -> str:
        # Implementation details go here
        return "Result from custom tool"
```

### Utilizing the `tool` Decorator

For a more straightforward approach, employ the `tool` decorator to create a `Tool` object directly. This method requires specifying the required attributes and functional logic within a decorated function.

```python
from crewai_tools import tool

@tool("Name of my tool")
def my_tool(question: str) -> str:
    """Provide a clear description of what this tool is useful for. Your agent will need this information to use it."""
    # Implement function logic here
```

```python
import json
import requests
from crewai import Agent
from crewai.tools import tool

# Decorate the function with the tool decorator from crewAI
@tool("Integration with a Given API")
def integration_tool(argument: str) -> str:
    """Details the integration process with a given API."""
    # Implementation details
    return "Results to be sent back to the agent"
```

### Defining a Cache Function for the Tool

By default, all tools have caching enabled, meaning that if a tool is called with the same arguments by any agent in the crew, it will return the same result. However, specific scenarios may require more tailored caching strategies. For these cases, use the `cache_function` attribute to assign a function that determines whether the result should be cached.

```python
@tool("Integration with a Given API")
def integration_tool(argument: str) -> str:
    """Integration with a given API."""
    # Implementation details
    return "Results to be sent back to the agent"

def cache_strategy(arguments: dict, result: str) -> bool:
    if result == "some_value":
        return True
    return False

integration_tool.cache_function = cache_strategy
```