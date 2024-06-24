# ComposioTool Documentation

## Description

This tools is a wrapper around the composio toolset and gives your agent access to a wide variety of tools from the composio SDK.

## Installation

To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install composio-core 
pip install 'crewai[tools]'
```

## Example

The following example demonstrates how to initialize the tool and execute a mathematical operation:

```python

from composio import Action

from crewai_tools.tools.composio_tool.composio_tool import ComposioTool

tool = ComposioTool.from_tool(
    tool=Action.MATHEMATICAL_CALCULATOR,
)
```

