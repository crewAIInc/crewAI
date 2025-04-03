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

## Real-World Examples

### Example 1: Research Assistant with Web Search Tool

```python
from crewai import Agent
from crewai_tools import SearchTool
from crewai.tools import ToolWithInstruction

search_tool = SearchTool()
search_with_instructions = ToolWithInstruction(
    tool=search_tool,
    instructions="""
    Use this tool ONLY for factual information that requires up-to-date data.
    ALWAYS verify information by searching multiple sources.
    DO NOT use this tool for speculative questions or opinions.
    """
)

research_agent = Agent(
    role="Research Analyst",
    goal="Provide accurate and well-sourced information",
    backstory="You are a meticulous research analyst with attention to detail and fact-checking.",
    tools=[search_with_instructions],
)
```

### Example 2: Data Scientist with Multiple Analysis Tools

```python
from crewai import Agent
from crewai_tools import PythonTool, DataVisualizationTool
from crewai.tools import ToolWithInstruction

# Python tool for data processing
python_tool = PythonTool()
python_with_instructions = ToolWithInstruction(
    tool=python_tool,
    instructions="""
    Use this tool for data cleaning, transformation, and statistical analysis.
    ALWAYS include comments in your code.
    DO NOT use this tool for creating visualizations.
    """
)

# Visualization tool
viz_tool = DataVisualizationTool()
viz_with_instructions = ToolWithInstruction(
    tool=viz_tool,
    instructions="""
    Use this tool ONLY for creating data visualizations.
    ALWAYS label axes and include titles in your charts.
    PREFER simple visualizations that clearly communicate the main insight.
    """
)

data_scientist = Agent(
    role="Data Scientist",
    goal="Analyze data and create insightful visualizations",
    backstory="You are an experienced data scientist who excels at finding patterns in data.",
    tools=[python_with_instructions, viz_with_instructions],
)
```

## How Instructions Are Presented to Agents

When an agent considers using a tool, the instructions are included in the tool's description. For example, a tool with instructions might appear to the agent like this:

```
Tool: search_web
Description: Search the web for information on a given topic.
Instructions: Use this tool ONLY for factual information that requires up-to-date data. 
ALWAYS verify information by searching multiple sources. 
DO NOT use this tool for speculative questions or opinions.
```

This clear presentation helps the agent understand when and how to use the tool appropriately.

## Dynamically Updating Instructions

You can update tool instructions dynamically during execution:

```python
# Create a tool with initial instructions
search_with_instructions = ToolWithInstruction(
    tool=search_tool,
    instructions="Initial instructions for tool usage"
)

# Later, update the instructions based on new requirements
search_with_instructions.update_instructions("Updated instructions for tool usage")
```

## Error Handling and Best Practices

### Validation

The `ToolWithInstruction` class includes validation to ensure instructions are not empty and don't exceed a maximum length. If you provide invalid instructions, a `ValueError` will be raised.

### Best Practices for Writing Instructions

1. **Be specific and clear** about when to use and when not to use the tool
2. **Use imperative language** like "ALWAYS", "NEVER", "USE", "DO NOT USE"
3. **Keep instructions concise** but comprehensive
4. **Include examples** of good and bad usage scenarios when possible
5. **Format instructions** with line breaks for readability

## When to Use Tool Instructions

Tool instructions are useful when:

1. You want to specify precise conditions for tool usage
2. You have multiple similar tools that should be used in different situations
3. You want to keep the agent's backstory focused on its role and personality, 
   not technical details about tools
4. You need to provide technical guidance on how to format inputs or interpret outputs
5. You want to enforce consistent tool usage across multiple agents

Tool instructions are semantically more correct than putting tool usage guidelines in the agent's backstory.
