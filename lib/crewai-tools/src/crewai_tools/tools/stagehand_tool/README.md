# Stagehand Web Automation Tool

This tool integrates the [Stagehand](https://docs.stagehand.dev/) framework with CrewAI, allowing agents to interact with websites and automate browser tasks using natural language instructions.

## Description

Stagehand is a powerful browser automation framework built by Browserbase that allows AI agents to:

- Navigate to websites
- Click buttons, links, and other elements
- Fill in forms
- Extract data from web pages
- Observe and identify elements
- Perform complex workflows

The StagehandTool wraps the Stagehand Python SDK to provide CrewAI agents with the ability to control a real web browser and interact with websites using three core primitives:

1. **Act**: Perform actions like clicking, typing, or navigating
2. **Extract**: Extract structured data from web pages
3. **Observe**: Identify and analyze elements on the page

## Requirements

Before using this tool, you will need:

1. A [Browserbase](https://www.browserbase.com/) account with API key and project ID
2. An API key for an LLM (OpenAI or Anthropic Claude)
3. The Stagehand Python SDK installed

Install the dependencies:

```bash
pip install stagehand-py
```

## Usage

### Basic Usage

The StagehandTool can be used in two ways:

1. **Using a context manager (recommended)**:
```python
from crewai import Agent, Task, Crew
from crewai_tools import StagehandTool
from stagehand.schemas import AvailableModel

# Initialize the tool with your API keys using a context manager
with StagehandTool(
    api_key="your-browserbase-api-key",
    project_id="your-browserbase-project-id",
    model_api_key="your-llm-api-key",  # OpenAI or Anthropic API key
    model_name=AvailableModel.CLAUDE_3_7_SONNET_LATEST,  # Optional: specify which model to use
) as stagehand_tool:
    # Create an agent with the tool
    researcher = Agent(
        role="Web Researcher",
        goal="Find and summarize information from websites",
        backstory="I'm an expert at finding information online.",
        verbose=True,
        tools=[stagehand_tool],
    )

    # Create a task that uses the tool
    research_task = Task(
        description="Go to https://www.example.com and tell me what you see on the homepage.",
        agent=researcher,
    )

    # Run the crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    result = crew.kickoff()
    print(result)
    # Resources are automatically cleaned up when exiting the context
```

2. **Manual resource management**:
```python
from crewai import Agent, Task, Crew
from crewai_tools import StagehandTool
from stagehand.schemas import AvailableModel

# Initialize the tool with your API keys
stagehand_tool = StagehandTool(
    api_key="your-browserbase-api-key",
    project_id="your-browserbase-project-id",
    model_api_key="your-llm-api-key",
    model_name=AvailableModel.CLAUDE_3_7_SONNET_LATEST,
)

try:
    # Create an agent with the tool
    researcher = Agent(
        role="Web Researcher",
        goal="Find and summarize information from websites",
        backstory="I'm an expert at finding information online.",
        verbose=True,
        tools=[stagehand_tool],
    )

    # Create a task that uses the tool
    research_task = Task(
        description="Go to https://www.example.com and tell me what you see on the homepage.",
        agent=researcher,
    )

    # Run the crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    result = crew.kickoff()
    print(result)
finally:
    # Explicitly clean up resources
    stagehand_tool.close()
```

The context manager approach (option 1) is recommended as it ensures proper cleanup of resources even if exceptions occur. However, both approaches are valid and will properly manage the browser session.

## Command Types

The StagehandTool supports three different command types, each designed for specific web automation tasks:

### 1. Act - Perform Actions on a Page

The `act` command type (default) allows the agent to perform actions on a webpage, such as clicking buttons, filling forms, navigating, and more.

**When to use**: Use `act` when you need to interact with a webpage by performing actions like clicking, typing, scrolling, or navigating.

**Example usage**:
```python
# Perform an action (default behavior)
result = stagehand_tool.run(
    instruction="Click the login button", 
    url="https://example.com",
    command_type="act"  # Default, so can be omitted
)

# Fill out a form
result = stagehand_tool.run(
    instruction="Fill the contact form with name 'John Doe', email 'john@example.com', and message 'Hello world'", 
    url="https://example.com/contact"
)

# Multiple actions in sequence
result = stagehand_tool.run(
    instruction="Search for 'AI tools' in the search box and press Enter", 
    url="https://example.com"
)
```

### 2. Extract - Get Data from a Page

The `extract` command type allows the agent to extract structured data from a webpage, such as product information, article text, or table data.

**When to use**: Use `extract` when you need to retrieve specific information from a webpage in a structured format.

**Example usage**:
```python
# Extract all product information
result = stagehand_tool.run(
    instruction="Extract all product names, prices, and descriptions", 
    url="https://example.com/products",
    command_type="extract"
)

# Extract specific information with a selector
result = stagehand_tool.run(
    instruction="Extract the main article title and content", 
    url="https://example.com/blog/article",
    command_type="extract",
    selector=".article-container"  # Optional CSS selector to limit extraction scope
)

# Extract tabular data
result = stagehand_tool.run(
    instruction="Extract the data from the pricing table as a structured list of plans with their features and costs", 
    url="https://example.com/pricing",
    command_type="extract",
    selector=".pricing-table"
)
```

### 3. Observe - Identify Elements on a Page

The `observe` command type allows the agent to identify and analyze specific elements on a webpage, returning information about their attributes, location, and suggested actions.

**When to use**: Use `observe` when you need to identify UI elements, understand page structure, or determine what actions are possible.

**Example usage**:
```python
# Find interactive elements
result = stagehand_tool.run(
    instruction="Find all interactive elements in the navigation menu", 
    url="https://example.com",
    command_type="observe"
)

# Identify form fields
result = stagehand_tool.run(
    instruction="Identify all the input fields in the registration form", 
    url="https://example.com/register",
    command_type="observe",
    selector="#registration-form"
)

# Analyze page structure
result = stagehand_tool.run(
    instruction="Find the main content sections of this page", 
    url="https://example.com/about",
    command_type="observe"
)
```

## Advanced Configuration

You can customize the behavior of the StagehandTool by specifying different parameters:

```python
stagehand_tool = StagehandTool(
    api_key="your-browserbase-api-key",
    project_id="your-browserbase-project-id",
    model_api_key="your-llm-api-key",
    model_name=AvailableModel.CLAUDE_3_7_SONNET_LATEST,
    dom_settle_timeout_ms=5000,  # Wait longer for DOM to settle
    headless=True,  # Run browser in headless mode (no visible window)
    self_heal=True,  # Attempt to recover from errors
    wait_for_captcha_solves=True,  # Wait for CAPTCHA solving
    verbose=1,  # Control logging verbosity (0-3)
)
```

## Tips for Effective Use

1. **Be specific in instructions**: The more specific your instructions, the better the results. For example, instead of "click the button," use "click the 'Submit' button at the bottom of the contact form."

2. **Use the right command type**: Choose the appropriate command type based on your task:
   - Use `act` for interactions and navigation
   - Use `extract` for gathering information
   - Use `observe` for understanding page structure

3. **Leverage selectors**: When extracting data or observing elements, use CSS selectors to narrow the scope and improve accuracy.

4. **Handle multi-step processes**: For complex workflows, break them down into multiple tool calls, each handling a specific step.

5. **Error handling**: Implement appropriate error handling in your agent's logic to deal with potential issues like elements not found or pages not loading.

## Troubleshooting

- **Session not starting**: Ensure you have valid API keys for both Browserbase and your LLM provider.
- **Elements not found**: Try increasing the `dom_settle_timeout_ms` parameter to give the page more time to load.
- **Actions not working**: Make sure your instructions are clear and specific. You may need to use `observe` first to identify the correct elements.
- **Extract returning incomplete data**: Try refining your instruction or providing a more specific selector.

## Resources

- [Stagehand Documentation](https://docs.stagehand.dev/reference/introduction) - Complete reference for the Stagehand framework
- [Browserbase](https://www.browserbase.com) - Browser automation platform
- [Join Slack Community](https://stagehand.dev/slack) - Get help and connect with other users of Stagehand

## Contact

For more information about Stagehand, visit [the Stagehand documentation](https://docs.stagehand.dev/).

For questions about the CrewAI integration, join our [Slack](https://stagehand.dev/slack) or open an issue in this repository.