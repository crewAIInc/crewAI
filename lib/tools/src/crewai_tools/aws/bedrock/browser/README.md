# AWS Bedrock Browser Tools

This toolkit provides a set of tools for interacting with web browsers through AWS Bedrock Browser. It enables your CrewAI agents to navigate websites, extract content, click elements, and more.

## Features

- Navigate to URLs and browse the web
- Extract text and hyperlinks from pages
- Click on elements using CSS selectors
- Navigate back through browser history
- Get information about the current webpage
- Multiple browser sessions with thread-based isolation

## Installation

Ensure you have the necessary dependencies:

```bash
uv add crewai-tools bedrock-agentcore beautifulsoup4 playwright nest-asyncio
```

## Usage

### Basic Usage

```python
from crewai import Agent, Task, Crew, LLM
from crewai_tools.aws.bedrock.browser import create_browser_toolkit

# Create the browser toolkit
toolkit, browser_tools = create_browser_toolkit(region="us-west-2")

# Create the Bedrock LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-west-2",
)

# Create a CrewAI agent that uses the browser tools
research_agent = Agent(
    role="Web Researcher",
    goal="Research and summarize web content",
    backstory="You're an expert at finding information online.",
    tools=browser_tools,
    llm=llm
)

# Create a task for the agent
research_task = Task(
    description="Navigate to https://example.com and extract all text content. Summarize the main points.",
    expected_output="A list of bullet points containing the most important information on https://example.com. Plus, a description of the tool calls used, and actions performed to get to the page.",
    agent=research_agent
)

# Create and run the crew
crew = Crew(
    agents=[research_agent],
    tasks=[research_task]
)
result = crew.kickoff()

print(f"\n***Final result:***\n\n{result}")

# Clean up browser resources when done
toolkit.sync_cleanup()
```

### Available Tools

The toolkit provides the following tools:

1. `navigate_browser` - Navigate to a URL
2. `click_element` - Click on an element using CSS selectors
3. `extract_text` - Extract all text from the current webpage
4. `extract_hyperlinks` - Extract all hyperlinks from the current webpage
5. `get_elements` - Get elements matching a CSS selector
6. `navigate_back` - Navigate to the previous page
7. `current_webpage` - Get information about the current webpage

### Advanced Usage (with async)

```python
import asyncio
from crewai import Agent, Task, Crew, LLM
from crewai_tools.aws.bedrock.browser import create_browser_toolkit

async def main():

    # Create the browser toolkit with specific AWS region
    toolkit, browser_tools = create_browser_toolkit(region="us-west-2")
    tools_by_name = toolkit.get_tools_by_name()

    # Create the Bedrock LLM
    llm = LLM(
        model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name="us-west-2",
    )

    # Create agents with specific tools
    navigator_agent = Agent(
        role="Navigator",
        goal="Find specific information across websites",
        backstory="You navigate through websites to locate information.",
        tools=[
            tools_by_name["navigate_browser"],
            tools_by_name["click_element"],
            tools_by_name["navigate_back"]
        ],
        llm=llm
    )

    content_agent = Agent(
        role="Content Extractor",
        goal="Extract and analyze webpage content",
        backstory="You extract and analyze content from webpages.",
        tools=[
            tools_by_name["extract_text"],
            tools_by_name["extract_hyperlinks"],
            tools_by_name["get_elements"]
        ],
        llm=llm
    )

    # Create tasks for the agents
    navigation_task = Task(
        description="Navigate to https://example.com, then click on the the 'More information...' link.",
        expected_output="The status of the tool calls for this task.",
        agent=navigator_agent,
    )

    extraction_task = Task(
        description="Extract all text from the current page and summarize it.",
        expected_output="The summary of the page, and a description of the tool calls used, and actions performed to get to the page.",
        agent=content_agent,
    )

    # Create and run the crew
    crew = Crew(
        agents=[navigator_agent, content_agent],
        tasks=[navigation_task, extraction_task]
    )

    result = await crew.kickoff_async()

    # Clean up browser resources when done
    toolkit.sync_cleanup()

    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\n***Final result:***\n\n{result}")
```

## Requirements

- AWS account with access to Bedrock AgentCore API
- Properly configured AWS credentials