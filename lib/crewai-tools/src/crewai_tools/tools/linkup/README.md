# Linkup Search Tool

## Description

The `LinkupSearchTool` is a tool designed for integration with the CrewAI framework. It provides the ability to query the Linkup API for contextual information and retrieve structured results. This tool is ideal for enriching workflows with up-to-date and reliable information from Linkup.

---

## Features

- Perform API queries to the Linkup platform using customizable parameters (`query`, `depth`, `output_type`).
- Gracefully handles API errors and provides structured feedback.
- Returns well-structured results for seamless integration into CrewAI processes.

---

## Installation

### Prerequisites

- Linkup API Key

### Steps

1. ```shell
  pip install 'crewai[tools]'
  ```

2. Create a `.env` file in your project root and add your Linkup API Key:
   ```plaintext
   LINKUP_API_KEY=your_linkup_api_key
   ```

---

## Usage

### Basic Example

Here is how to use the `LinkupSearchTool` in a CrewAI project:

1. **Import and Initialize**:
   ```python
   from tools.linkup_tools import LinkupSearchTool
   import os
   from dotenv import load_dotenv

   load_dotenv()

   linkup_tool = LinkupSearchTool(api_key=os.getenv("LINKUP_API_KEY"))
   ```

2. **Set Up an Agent and Task**:
   ```python
   from crewai import Agent, Task, Crew

   # Define the agent
   research_agent = Agent(
       role="Information Researcher",
       goal="Fetch relevant results from Linkup.",
       backstory="An expert in online information retrieval...",
       tools=[linkup_tool],
       verbose=True
   )

   # Define the task
   search_task = Task(
       expected_output="A detailed list of Nobel Prize-winning women in physics with their achievements.",
       description="Search for women who have won the Nobel Prize in Physics.",
       agent=research_agent
   )

   # Create and run the crew
   crew = Crew(
       agents=[research_agent],
       tasks=[search_task]
   )

   result = crew.kickoff()
   print(result)
   ```

### Advanced Configuration

You can customize the parameters for the `LinkupSearchTool`:

- `query`: The search term or phrase.
- `depth`: The search depth (`"standard"` by default).
- `output_type`: The type of output (`"searchResults"` by default).

Example:
```python
response = linkup_tool._run(
    query="Women Nobel Prize Physics",
    depth="standard",
    output_type="searchResults"
)
```