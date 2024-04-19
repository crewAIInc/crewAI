---
title: Creating and Utilizing Tools in crewAI
description: Comprehensive guide on crafting, using, and managing custom tools within the crewAI framework, including new functionalities and error handling.
---

## Creating and Utilizing Tools in crewAI
This guide provides detailed instructions on creating custom tools for the crewAI framework and how to efficiently manage and utilize these tools, incorporating the latest functionalities such as tool delegation, error handling, and dynamic tool calling. It also highlights the importance of collaboration tools, enabling agents to perform a wide range of actions.

### Prerequisites
Before creating your own tools, ensure you have the crewAI extra tools package installed:

```bash
pip install 'crewai[tools]'
```

### Subclassing `BaseTool`

To create a personalized tool, inherit from `BaseTool` and define the necessary attributes and the `_run` method.

```python
from crewai_tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "What this tool does. It's vital for effective utilization."

    def _run(self, argument: str) -> str:
        # Your tool's logic here
        return "Tool's result"
```

### Using the `tool` Decorator

Alternatively, use the `tool` decorator for a direct approach to create tools. This requires specifying attributes and the tool's logic within a function.

```python
from crewai_tools import tool

@tool("Tool Name")
def my_simple_tool(question: str) -> str:
    """Tool description for clarity."""
    # Tool logic here
    return "Tool output"
```
### Defining a Cache Function for the Tool

To optimize tool performance with caching, define custom caching strategies using the `cache_function` attribute.

```python
@tool("Tool with Caching")
def cached_tool(argument: str) -> str:
    """Tool functionality description."""
    return "Cachable result"

def my_cache_strategy(arguments: dict, result: str) -> bool:
    # Define custom caching logic
    return True if some_condition else False

cached_tool.cache_function = my_cache_strategy
```

By adhering to these guidelines and incorporating new functionalities and collaboration tools into your tool creation and management processes, you can leverage the full capabilities of the crewAI framework, enhancing both the development experience and the efficiency of your AI agents.