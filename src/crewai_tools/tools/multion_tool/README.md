# MultiOnTool Documentation

## Description
The MultiOnTool, integrated within the crewai_tools package, empowers CrewAI agents with the capability to navigate and interact with the web through natural language instructions. Leveraging the Multion API, this tool facilitates seamless web browsing, making it an essential asset for projects requiring dynamic web data interaction.

## Installation
Ensure the `crewai[tools]` package is installed in your environment to use the MultiOnTool. If it's not already installed, you can add it using the command below:

## Example
The following example demonstrates how to initialize the tool and execute a search with a given query:

```python
from crewai_tools import MultiOnTool

# Initialize the tool from a MultiOn Tool
multion_tool = MultiOnTool(api_key= "YOUR_MULTION_API_KEY", local=False)

```

## Arguments

- `api_key`: Specifies Browserbase API key. Defaults is the `BROWSERBASE_API_KEY` environment variable.
- `local`: Optional. Use the local flag set as "true" to run the agent locally on your browser. Make sure the multion browser extension is installed and API Enabled is checked.

## Steps to Get Started
To effectively use the `MultiOnTool`, follow these steps:

1. **Install CrewAI**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **Install and use MultiOn**: Follow MultiOn documentation (https://docs.multion.ai/).


