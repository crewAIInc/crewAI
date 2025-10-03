# Zapier Action Tools

## Description

This tool enables CrewAI agents to interact with Zapier actions, allowing them to automate workflows and integrate with hundreds of applications through Zapier's platform. The tool dynamically creates BaseTool instances for each available Zapier action, making it easy to incorporate automation into your AI workflows.

## Installation

Install the crewai_tools package by executing the following command in your terminal:

```shell
uv pip install 'crewai[tools]'
```

## Example

To utilize the ZapierActionTools for different use cases, follow these examples:

```python
from crewai_tools import ZapierActionTools
from crewai import Agent

# Get all available Zapier actions you are connected to.
tools = ZapierActionTools(
    zapier_api_key="your-zapier-api-key"
)

# Or specify only certain actions you want to use
tools = ZapierActionTools(
    zapier_api_key="your-zapier-api-key",
    action_list=["gmail_find_email", "slack_send_message", "google_sheets_create_row"]
)

# Adding the tools to an agent
zapier_agent = Agent(
    name="zapier_agent",
    role="You are a helpful assistant that can automate tasks using Zapier integrations.",
    llm="gpt-4o-mini",
    tools=tools,
    goal="Automate workflows and integrate with various applications",
    backstory="You are a Zapier automation expert that helps users connect and automate their favorite apps.",
    verbose=True,
)

# Example usage
result = zapier_agent.kickoff(
    "Find emails from john@example.com in Gmail"
)
```

## Arguments

- `zapier_api_key` : Your Zapier API key for authentication. Can also be set via `ZAPIER_API_KEY` environment variable. (Required)
- `action_list` : A list of specific Zapier action names to include. If not provided, all available actions will be returned. (Optional)

## Environment Variables

You can set your Zapier API key as an environment variable instead of passing it directly:

```bash
export ZAPIER_API_KEY="your-zapier-api-key"
```

Then use the tool without explicitly passing the API key:

```python
from crewai_tools import ZapierActionTools

# API key will be automatically loaded from environment
tools = ZapierActionTools(
    action_list=["gmail_find_email", "slack_send_message"]
)
```

## Getting Your Zapier API Key

1. Log in to your Zapier account
2. Go to https://zapier.com/app/developer/
3. Create a new app or use an existing one
4. Navigate to the "Authentication" section
5. Copy your API key

## Available Actions

The tool will dynamically discover all available Zapier actions associated with your API key. Common actions include:

- Gmail operations (find emails, send emails)
- Slack messaging
- Google Sheets operations
- Calendar events
- And hundreds more depending on your Zapier integrations
