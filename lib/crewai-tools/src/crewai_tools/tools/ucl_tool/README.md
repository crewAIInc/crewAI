# UCLTool Documentation

## Description

UCLTool is a wrapper around the UCL (Unified Context Layer) API that gives your CrewAI agents access to a wide variety of tools and integrations through the Fastn.ai platform. UCL provides a unified interface to connect with multiple services and APIs.

## Installation

To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install requests
pip install 'crewai[tools]'
```

## Configuration

Before using UCLTool, you need to obtain the following credentials from your ucl.dev workspace:

- **UCL_WORKSPACE_ID**: Your UCL workspace identifier
- **UCL_API_KEY**: Your workspace API key
- **UCL_MCP_GATEWAY_ID**: Your workspace MCP Gateway identifier

You can either set these as environment variables or pass them directly to the configuration.

### Environment Variables

```shell
export UCL_WORKSPACE_ID="your-workspace-id"
export UCL_API_KEY="your-api-key"
export UCL_MCP_GATEWAY_ID="your-mcp-gateway-id"
```

## Example

The following example demonstrates how to initialize the tool and use UCL actions with CrewAI:

### 1. Create UCL Configuration

```python
from crewai_tools import UCLTool

# Create configuration
config = UCLTool.from_config(
    workspace_id="your-workspace-id",
    api_key="your-api-key",
    mcp_gateway_id="your-mcp-gateway-id",
)
```

### 2. Get Available Tools

You can fetch all available tools or filter them by a prompt:

```python
# Get all available tools (up to 500)
tools = UCLTool.get_tools(config)

# Get tools related to specific keywords
tools = UCLTool.get_tools(config, prompt="slack messaging", limit=10)
```

### 3. Create a Specific Tool

If you know the specific action you want to use:

```python
tool = UCLTool.from_action(
    config=config,
    action_id="your-action-id",
    tool_name="send_slack_message",
    description="Send a message to a Slack channel",
    input_schema={
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Slack channel name"},
            "message": {"type": "string", "description": "Message content"},
        },
        "required": ["channel", "message"],
    },
)
```

### 4. Define Agent with UCL Tools

```python
from crewai import Agent, Task

# Get tools for a specific use case
tools = UCLTool.get_tools(config, prompt="email and calendar", limit=20)

# Create an agent with UCL tools
agent = Agent(
    role="Communication Assistant",
    goal="Help users manage their communications and scheduling",
    backstory=(
        "You are an AI assistant that helps users send emails, "
        "manage calendar events, and handle communications across "
        "various platforms using UCL integrations."
    ),
    verbose=True,
    tools=tools,
)
```

### 5. Execute Task

```python
task = Task(
    description="Send a welcome email to new-user@example.com with subject 'Welcome to our platform'",
    agent=agent,
    expected_output="Confirmation that the email was sent successfully",
)

result = task.execute()
print(result)
```

## Complete Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import UCLTool

# Step 1: Configure UCL
config = UCLTool.from_config(
    workspace_id="your-workspace-id",
    api_key="your-api-key",
    mcp_gateway_id="your-mcp-gateway-id",
)

# Step 2: Get relevant tools
tools = UCLTool.get_tools(config, prompt="slack notifications", limit=10)

# Step 3: Create agent
notification_agent = Agent(
    role="Notification Manager",
    goal="Send notifications to team members via Slack",
    backstory="You are responsible for keeping the team informed about important updates.",
    verbose=True,
    tools=tools,
)

# Step 4: Define task
task = Task(
    description="Send a notification to the #general channel about the deployment completion",
    agent=notification_agent,
    expected_output="Confirmation of notification sent",
)

# Step 5: Create and run crew
crew = Crew(
    agents=[notification_agent],
    tasks=[task],
    verbose=True,
)

result = crew.kickoff()
print(result)
```

## API Reference

### UCLTool.from_config()

Creates a UCL configuration object.

**Parameters:**
- `workspace_id` (str): UCL Workspace ID
- `api_key` (str): Workspace API Key
- `mcp_gateway_id` (str): Workspace MCP Gateway ID
- `base_url` (str, optional): Base URL for UCL API. Default: `https://live.fastn.ai`
- `stage` (str, optional): API stage. Default: `LIVE`

**Returns:** `UCLToolConfig`

### UCLTool.get_tools()

Fetches and creates tools from the UCL API.

**Parameters:**
- `config` (UCLToolConfig): UCL configuration object
- `prompt` (str, optional): Filter tools by keywords
- `limit` (int, optional): Maximum number of tools to fetch. Default: 500

**Returns:** `list[UCLTool]`

### UCLTool.from_action()

Creates a UCLTool from a specific action.

**Parameters:**
- `config` (UCLToolConfig): UCL configuration object
- `action_id` (str): UCL Action ID
- `tool_name` (str): Name of the tool
- `description` (str): Tool description
- `input_schema` (dict): JSON schema for tool input parameters

**Returns:** `UCLTool`

## More Information

For more detailed information about UCL and available integrations, visit [ucl.dev](https://docs.ucl.dev/getting-started/about-unified-context-layer).
