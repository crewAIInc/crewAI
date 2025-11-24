# MergeAgentHandlerTool Documentation

## Description

This tool is a wrapper around the Merge Agent Handler platform and gives your agent access to third-party tools and integrations via the Model Context Protocol (MCP). Merge Agent Handler securely manages authentication, permissions, and monitoring of all tool interactions across platforms like Linear, Jira, Slack, GitHub, and many more.

## Installation

### Step 1: Set up a virtual environment (recommended)

It's recommended to use a virtual environment to avoid conflicts with other packages:

```shell
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 2: Install CrewAI Tools

To incorporate this tool into your project, install CrewAI with tools support:

```shell
pip install 'crewai[tools]'
```

### Step 3: Set up your Agent Handler credentials

You'll need to set up your Agent Handler API key. You can get your API key from the [Agent Handler dashboard](https://ah.merge.dev).

```shell
# Set the API key in your current terminal session
export AGENT_HANDLER_API_KEY='your-api-key-here'

# Or add it to your shell profile for persistence (e.g., ~/.bashrc, ~/.zshrc)
echo "export AGENT_HANDLER_API_KEY='your-api-key-here'" >> ~/.zshrc
source ~/.zshrc
```

**Alternative: Use a `.env` file**

You can also use a `.env` file in your project directory:

```shell
# Create a .env file
echo "AGENT_HANDLER_API_KEY=your-api-key-here" > .env

# Load it in your Python script
from dotenv import load_dotenv
load_dotenv()
```

**Note**: Make sure to add `.env` to your `.gitignore` to avoid committing secrets!

## Prerequisites

Before using this tool, you need to:

1. **Create a Tool Pack** in Agent Handler with the connectors and tools you want to use
2. **Register a User** who will be executing the tools
3. **Authenticate connectors** for the registered user (using Agent Handler Link)

You can do this via the [Agent Handler dashboard](https://ah.merge.dev) or the [Agent Handler API](https://docs.ah.merge.dev).

## Example Usage

### Example 1: Using a specific tool

The following example demonstrates how to initialize a specific tool and use it with a CrewAI agent:

```python
from crewai_tools import MergeAgentHandlerTool
from crewai import Agent, Task

# Initialize a specific tool
create_issue_tool = MergeAgentHandlerTool.from_tool_name(
    tool_name="linear__create_issue",
    tool_pack_id="134e0111-0f67-44f6-98f0-597000290bb3",
    registered_user_id="91b2b905-e866-40c8-8be2-efe53827a0aa"
)

# Define agent with the tool
project_manager = Agent(
    role="Project Manager",
    goal="Create and manage project tasks efficiently",
    backstory=(
        "You are an experienced project manager who tracks tasks "
        "and issues across various project management tools."
    ),
    verbose=True,
    tools=[create_issue_tool],
)

# Execute task
task = Task(
    description="Create a new issue in Linear titled 'Implement user authentication' with high priority",
    agent=project_manager,
    expected_output="Confirmation that the issue was created with its ID",
)

task.execute()
```

### Example 2: Loading all tools from a Tool Pack

You can load all tools from a Tool Pack at once:

```python
from crewai_tools import MergeAgentHandlerTool
from crewai import Agent, Task

# Load all tools from a Tool Pack
tools = MergeAgentHandlerTool.from_tool_pack(
    tool_pack_id="134e0111-0f67-44f6-98f0-597000290bb3",
    registered_user_id="91b2b905-e866-40c8-8be2-efe53827a0aa"
)

# Define agent with all tools
support_agent = Agent(
    role="Support Engineer",
    goal="Handle customer support requests across multiple platforms",
    backstory=(
        "You are a skilled support engineer who can access customer "
        "data and create tickets across various support tools."
    ),
    verbose=True,
    tools=tools,
)
```

### Example 3: Loading specific tools from a Tool Pack

You can also load only specific tools from a Tool Pack:

```python
from crewai_tools import MergeAgentHandlerTool

# Load only specific tools
tools = MergeAgentHandlerTool.from_tool_pack(
    tool_pack_id="134e0111-0f67-44f6-98f0-597000290bb3",
    registered_user_id="91b2b905-e866-40c8-8be2-efe53827a0aa",
    tool_names=["linear__create_issue", "linear__get_issues", "slack__send_message"]
)
```

### Example 4: Using with local/staging environment

For development, you can point to a different Agent Handler environment:

```python
from crewai_tools import MergeAgentHandlerTool

# Use with local or staging environment
tool = MergeAgentHandlerTool.from_tool_name(
    tool_name="linear__create_issue",
    tool_pack_id="your-tool-pack-id",
    registered_user_id="your-user-id",
    base_url="http://localhost:8000"  # or your staging URL
)
```

## API Reference

### Class Methods

#### `from_tool_name()`

Create a single tool instance for a specific tool.

**Parameters:**
- `tool_name` (str): Name of the tool (e.g., "linear__create_issue")
- `tool_pack_id` (str): UUID of the Tool Pack
- `registered_user_id` (str): UUID or origin_id of the registered user
- `base_url` (str, optional): Base URL for Agent Handler API (defaults to "https://api.ah.merge.dev")

**Returns:** `MergeAgentHandlerTool` instance

#### `from_tool_pack()`

Create multiple tool instances from a Tool Pack.

**Parameters:**
- `tool_pack_id` (str): UUID of the Tool Pack
- `registered_user_id` (str): UUID or origin_id of the registered user
- `tool_names` (List[str], optional): List of specific tool names to load. If None, loads all tools.
- `base_url` (str, optional): Base URL for Agent Handler API (defaults to "https://api.ah.merge.dev")

**Returns:** `List[MergeAgentHandlerTool]` instances

## Available Connectors

Merge Agent Handler supports 100+ integrations including:

**Project Management:** Linear, Jira, Asana, Monday, ClickUp, Height, Shortcut

**Communication:** Slack, Microsoft Teams, Discord

**CRM:** Salesforce, HubSpot, Pipedrive

**Development:** GitHub, GitLab, Bitbucket

**Documentation:** Notion, Confluence, Google Docs

**And many more...**

For a complete list of available connectors and tools, visit the [Agent Handler documentation](https://docs.ah.merge.dev).

## Authentication

Agent Handler handles all authentication for you. Users authenticate to third-party services via Agent Handler Link, and the platform securely manages tokens and credentials. Your agents can then execute tools without worrying about authentication details.

## Security

All tool executions are:
- **Logged and monitored** for audit trails
- **Scanned for PII** to prevent sensitive data leaks
- **Rate limited** based on your plan
- **Permission-controlled** at the user and organization level

## Support

For questions or issues:
- 📚 [Documentation](https://docs.ah.merge.dev)
- 💬 [Discord Community](https://merge.dev/discord)
- 📧 [Support Email](mailto:support@merge.dev)
