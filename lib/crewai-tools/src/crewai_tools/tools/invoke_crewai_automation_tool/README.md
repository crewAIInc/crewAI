# InvokeCrewAIAutomationTool

## Description

The InvokeCrewAIAutomationTool provides CrewAI Platform API integration with external crew services. This tool allows you to invoke and interact with CrewAI Platform automations from within your CrewAI agents, enabling seamless integration between different crew workflows.

## Features

- **Dynamic Input Schema**: Configure custom input parameters for different crew automations
- **Automatic Polling**: Automatically polls for task completion with configurable timeout
- **Bearer Token Authentication**: Secure API authentication using bearer tokens
- **Comprehensive Error Handling**: Robust error handling for API failures and timeouts
- **Flexible Configuration**: Support for both simple and complex crew automation workflows

## Installation

Install the required dependencies:

```shell
pip install 'crewai[tools]'
```

## Example

### Basic Usage

```python
from crewai_tools import InvokeCrewAIAutomationTool

# Basic crew automation tool
tool = InvokeCrewAIAutomationTool(
    crew_api_url="https://data-analysis-crew-[...].crewai.com",
    crew_bearer_token="your_bearer_token_here",
    crew_name="Data Analysis Crew",
    crew_description="Analyzes data and generates insights"
)

# Use the tool
result = tool.run()
```

### Advanced Usage with Custom Inputs

```python
from crewai_tools import InvokeCrewAIAutomationTool
from pydantic import Field

# Define custom input schema
custom_inputs = {
    "year": Field(..., description="Year to retrieve the report for (integer)"),
    "region": Field(default="global", description="Geographic region for analysis"),
    "format": Field(default="summary", description="Report format (summary, detailed, raw)")
}

# Create tool with custom inputs
tool = InvokeCrewAIAutomationTool(
    crew_api_url="https://state-of-ai-report-crew-[...].crewai.com",
    crew_bearer_token="your_bearer_token_here",
    crew_name="State of AI Report",
    crew_description="Retrieves a comprehensive report on state of AI for a given year and region",
    crew_inputs=custom_inputs,
    max_polling_time=15 * 60  # 15 minutes timeout
)

# Use with custom parameters
result = tool.run(year=2024, region="north-america", format="detailed")
```

### Integration with CrewAI Agents

```python
from crewai import Agent, Task, Crew
from crewai_tools import InvokeCrewAIAutomationTool

# Create the automation tool
market_research_tool = InvokeCrewAIAutomationTool(
    crew_api_url="https://market-research-automation-crew-[...].crewai.com",
    crew_bearer_token="your_bearer_token_here",
    crew_name="Market Research Automation",
    crew_description="Conducts comprehensive market research analysis",
    inputs={
        "year": Field(..., description="Year to use for the market research"),
    }
)

# Create an agent with the tool
research_agent = Agent(
    role="Research Coordinator",
    goal="Coordinate and execute market research tasks",
    backstory="You are an expert at coordinating research tasks and leveraging automation tools.",
    tools=[market_research_tool],
    verbose=True
)

# Create and execute a task
research_task = Task(
    description="Conduct market research on AI tools market for 2024",
    agent=research_agent,
    expected_output="Comprehensive market research report"
)

crew = Crew(
    agents=[research_agent],
    tasks=[research_task]
)

result = crew.kickoff()
```

## Arguments

### Required Parameters

- `crew_api_url` (str): Base URL of the CrewAI Platform automation API
- `crew_bearer_token` (str): Bearer token for API authentication
- `crew_name` (str): Name of the crew automation
- `crew_description` (str): Description of what the crew automation does

### Optional Parameters

- `max_polling_time` (int): Maximum time in seconds to wait for task completion (default: 600 seconds = 10 minutes)
- `crew_inputs` (dict): Dictionary defining custom input schema fields using Pydantic Field objects

## Custom Input Schema

When defining `crew_inputs`, use Pydantic Field objects to specify the input parameters. These have to be compatible with the crew automation you are invoking:

```python
from pydantic import Field

crew_inputs = {
    "required_param": Field(..., description="This parameter is required"),
    "optional_param": Field(default="default_value", description="This parameter is optional"),
    "typed_param": Field(..., description="Integer parameter", ge=1, le=100)  # With validation
}
```

## Error Handling

The tool provides comprehensive error handling for common scenarios:

- **API Connection Errors**: Network connectivity issues
- **Authentication Errors**: Invalid or expired bearer tokens
- **Timeout Errors**: Tasks that exceed the maximum polling time
- **Task Failures**: Crew automations that fail during execution

## API Endpoints

The tool interacts with two main API endpoints:

- `POST {crew_api_url}/kickoff`: Starts a new crew automation task
- `GET {crew_api_url}/status/{crew_id}`: Checks the status of a running task

## Notes

- The tool automatically polls the status endpoint every second until completion or timeout
- Successful tasks return the result directly, while failed tasks return error information
- The bearer token should be kept secure and not hardcoded in production environments
- Consider using environment variables for sensitive configuration like bearer tokens