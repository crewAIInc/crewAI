# ScalekitTool Documentation

## Description

This tool is a wrapper around the [Scalekit](https://scalekit.com) AgentKit SDK and gives your agent access to OAuth-authenticated tools across 50+ providers (Gmail, Slack, GitHub, Notion, Salesforce, and more).

Scalekit handles OAuth token management, refresh flows, and connected account lifecycle so your agents can focus on executing actions.

## Installation

To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install scalekit-sdk-python
pip install 'crewai[tools]'
```

After installation, set your Scalekit credentials as environment variables. Get these from [Scalekit Dashboard](https://app.scalekit.com) → **Developers** → **API Credentials**:

```bash
export SCALEKIT_ENV_URL="https://your-project.scalekit.cloud"
export SCALEKIT_CLIENT_ID="skc_..."
export SCALEKIT_CLIENT_SECRET="test_..."
```

## Example

The following example demonstrates how to initialize the tool and execute Gmail actions:

### 1. Create tools from a connection

```python
from crewai_tools import ScalekitTool
from crewai import Agent, Task

# Get all Gmail tools for a user
tools = ScalekitTool.from_connection(
    "gmail",
    identifier="user@example.com",
)
```

Or create a single specific tool:

```python
draft_tool = ScalekitTool.from_tool(
    tool_name="gmail_create_draft",
    identifier="user@example.com",
)
```

### 2. Define agent

```python
agent = Agent(
    role="Email Assistant",
    goal="Help manage emails and draft responses",
    backstory=(
        "You are an AI agent that helps users manage their email. "
        "You can read, draft, and organize emails on behalf of users."
    ),
    verbose=True,
    tools=tools,
)
```

### 3. Execute task

```python
task = Task(
    description="Draft a reply to the latest unread email",
    agent=agent,
    expected_output="Confirmation that the draft was created",
)

task.execute()
```

### Multiple providers

You can combine tools from multiple connections:

```python
tools = ScalekitTool.from_connection(
    "gmail", "slack", "github-prod",
    identifier="user@example.com",
)
```

## Connected Accounts

Before using ScalekitTool, ensure the user has connected their accounts through Scalekit. Use the Scalekit Dashboard → **AgentKit** → **Connections** to set up OAuth connections for providers your agents need.

* More details at [Scalekit AgentKit Docs](https://docs.scalekit.com)
