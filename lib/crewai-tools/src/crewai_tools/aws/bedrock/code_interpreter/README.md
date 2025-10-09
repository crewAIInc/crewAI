# AWS Bedrock Code Interpreter Tools

This toolkit provides a set of tools for interacting with the AWS Bedrock Code Interpreter environment. It enables your CrewAI agents to execute code, run shell commands, manage files, and perform computational tasks in a secure, isolated environment.

## Features

- Execute code in various languages (primarily Python)
- Run shell commands in the environment
- Read, write, list, and delete files
- Manage long-running tasks asynchronously
- Multiple code interpreter sessions with thread-based isolation

## Installation

Ensure you have the necessary dependencies:

```bash
uv add crewai-tools bedrock-agentcore
```

## Usage

### Basic Usage

```python
from crewai import Agent, Task, Crew, LLM
from crewai_tools.aws import create_code_interpreter_toolkit

# Create the code interpreter toolkit
toolkit, code_tools = create_code_interpreter_toolkit(region="us-west-2")

# Create the Bedrock LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-west-2",
)

# Create a CrewAI agent that uses the code interpreter tools
developer_agent = Agent(
    role="Python Developer",
    goal="Create and execute Python code to solve problems.",
    backstory="You're a skilled Python developer with expertise in data analysis.",
    tools=code_tools,
    llm=llm
)

# Create a task for the agent
coding_task = Task(
    description="Write a Python function that calculates the factorial of a number and test it. Do not use any imports from outside the Python standard library.",
    expected_output="The Python function created, and the test results.",
    agent=developer_agent
)

# Create and run the crew
crew = Crew(
    agents=[developer_agent],
    tasks=[coding_task]
)
result = crew.kickoff()

print(f"\n***Final result:***\n\n{result}")

# Clean up resources when done
import asyncio
asyncio.run(toolkit.cleanup())
```

### Available Tools

The toolkit provides the following tools:

1. `execute_code` - Run code in various languages (primarily Python)
2. `execute_command` - Run shell commands in the environment
3. `read_files` - Read content of files in the environment
4. `list_files` - List files in directories
5. `delete_files` - Remove files from the environment
6. `write_files` - Create or update files
7. `start_command_execution` - Start long-running commands asynchronously
8. `get_task` - Check status of async tasks
9. `stop_task` - Stop running tasks

### Advanced Usage

```python
from crewai import Agent, Task, Crew, LLM
from crewai_tools.aws import create_code_interpreter_toolkit

# Create the code interpreter toolkit
toolkit, code_tools = create_code_interpreter_toolkit(region="us-west-2")
tools_by_name = toolkit.get_tools_by_name()

# Create the Bedrock LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-west-2",
)

# Create agents with specific tools
code_agent = Agent(
    role="Code Developer",
    goal="Write and execute code",
    backstory="You write and test code to solve complex problems.",
    tools=[
        # Use specific tools by name
        tools_by_name["execute_code"],
        tools_by_name["execute_command"],
        tools_by_name["read_files"],
        tools_by_name["write_files"]
    ],
    llm=llm
)

file_agent = Agent(
    role="File Manager",
    goal="Manage files in the environment",
    backstory="You help organize and manage files in the code environment.",
    tools=[
        # Use specific tools by name
        tools_by_name["list_files"],
        tools_by_name["read_files"],
        tools_by_name["write_files"],
        tools_by_name["delete_files"]
    ],
    llm=llm
)

# Create tasks for the agents
coding_task = Task(
    description="Write a Python script to analyze data from a CSV file. Do not use any imports from outside the Python standard library.",
    expected_output="The Python function created.",
    agent=code_agent
)

file_task = Task(
    description="Organize the created files into separate directories.",
    agent=file_agent
)

# Create and run the crew
crew = Crew(
    agents=[code_agent, file_agent],
    tasks=[coding_task, file_task]
)
result = crew.kickoff()

print(f"\n***Final result:***\n\n{result}")

# Clean up code interpreter resources when done
import asyncio
asyncio.run(toolkit.cleanup())
```

### Example: Data Analysis with Python

```python
from crewai import Agent, Task, Crew, LLM
from crewai_tools.aws import create_code_interpreter_toolkit

# Create toolkit and tools
toolkit, code_tools = create_code_interpreter_toolkit(region="us-west-2")

# Create the Bedrock LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-west-2",
)

# Create a data analyst agent
analyst_agent = Agent(
    role="Data Analyst",
    goal="Analyze data using Python",
    backstory="You're an expert data analyst who uses Python for data processing.",
    tools=code_tools,
    llm=llm
)

# Create a task for the agent
analysis_task = Task(
    description="""
    For all of the below, do not use any imports from outside the Python standard library.
    1. Create a sample dataset with random data
    2. Perform statistical analysis on the dataset
    3. Generate visualizations of the results
    4. Save the results and visualizations to files
    """,
    agent=analyst_agent
)

# Create and run the crew
crew = Crew(
    agents=[analyst_agent],
    tasks=[analysis_task]
)
result = crew.kickoff()

print(f"\n***Final result:***\n\n{result}")

# Clean up resources
import asyncio
asyncio.run(toolkit.cleanup())
```

## Resource Cleanup

Always clean up code interpreter resources when done to prevent resource leaks:

```python
import asyncio

# Clean up all code interpreter sessions
asyncio.run(toolkit.cleanup())
```

## Requirements

- AWS account with access to Bedrock AgentCore API
- Properly configured AWS credentials