# Tool Instructions

CrewAI allows you to provide specific instructions for when and how to use tools. This is useful when you want to guide agents on proper tool usage without cluttering their backstory.

## Basic Usage

```python
from crewai import Agent
from crewai_tools import ScrapeWebsiteTool
from crewai.tools import ToolWithInstruction

# Create a tool with instructions
scrape_tool = ScrapeWebsiteTool()
scrape_with_instructions = ToolWithInstruction(
    tool=scrape_tool,
    instructions="""
    ALWAYS use this tool when making a joke.
    NEVER use this tool when making joke about someone's mom.
    """
)

# Use the tool with an agent
agent = Agent(
    role="Comedian",
    goal="Create hilarious and engaging jokes",
    backstory="""
          You are a professional stand-up comedian with years of experience in crafting jokes.
          You have a great sense of humor and can create jokes about any topic 
          while keeping them appropriate and entertaining.
    """,
    tools=[scrape_with_instructions],
)
```

## When to Use Tool Instructions

Tool instructions are useful when:

1. You want to specify precise conditions for tool usage
2. You have multiple similar tools that should be used in different situations
3. You want to keep the agent's backstory focused on its role and personality, 
   not technical details about tools

Tool instructions are semantically more correct than putting tool usage guidelines in the agent's backstory.
