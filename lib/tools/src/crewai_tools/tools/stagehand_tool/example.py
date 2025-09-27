"""
StagehandTool Example

This example demonstrates how to use the StagehandTool in a CrewAI workflow.
It shows how to use the three main primitives: act, extract, and observe.

Prerequisites:
1. A Browserbase account with API key and project ID
2. An LLM API key (OpenAI or Anthropic)
3. Installed dependencies: crewai, crewai-tools, stagehand-py

Usage:
- Set your API keys in environment variables (recommended)
- Or modify the script to include your API keys directly
- Run the script: python stagehand_example.py
"""

import os

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from stagehand.schemas import AvailableModel

from crewai_tools import StagehandTool

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
# You can set these in your shell or in a .env file
browserbase_api_key = os.environ.get("BROWSERBASE_API_KEY")
browserbase_project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
model_api_key = os.environ.get("OPENAI_API_KEY")  # or OPENAI_API_KEY

# Initialize the StagehandTool with your credentials and use context manager
with StagehandTool(
    api_key=browserbase_api_key,  # New parameter naming
    project_id=browserbase_project_id,  # New parameter naming
    model_api_key=model_api_key,
    model_name=AvailableModel.GPT_4O,  # Using the enum from schemas
) as stagehand_tool:
    # Create a web researcher agent with the StagehandTool
    researcher = Agent(
        role="Web Researcher",
        goal="Find and extract information from websites using different Stagehand primitives",
        backstory=(
            "You are an expert web automation agent equipped with the StagehandTool. "
            "Your primary function is to interact with websites based on natural language instructions. "
            "You must carefully choose the correct command (`command_type`) for each task:\n"
            "- Use 'act' (the default) for general interactions like clicking buttons ('Click the login button'), "
            "filling forms ('Fill the form with username user and password pass'), scrolling, or navigating within the site.\n"
            "- Use 'navigate' specifically when you need to go to a new web page; you MUST provide the target URL "
            "in the `url` parameter along with the instruction (e.g., instruction='Go to Google', url='https://google.com').\n"
            "- Use 'extract' when the goal is to pull structured data from the page. Provide a clear `instruction` "
            "describing what data to extract (e.g., 'Extract all product names and prices').\n"
            "- Use 'observe' to identify and analyze elements on the current page based on an `instruction` "
            "(e.g., 'Find all images in the main content area').\n\n"
            "Remember to break down complex tasks into simple, sequential steps in your `instruction`. For example, "
            "instead of 'Search for OpenAI on Google and click the first result', use multiple steps with the tool:\n"
            "1. Use 'navigate' with url='https://google.com'.\n"
            "2. Use 'act' with instruction='Type OpenAI in the search bar'.\n"
            "3. Use 'act' with instruction='Click the search button'.\n"
            "4. Use 'act' with instruction='Click the first search result link for OpenAI'.\n\n"
            "Always be precise in your instructions and choose the most appropriate command and parameters (`instruction`, `url`, `command_type`, `selector`) for the task at hand."
        ),
        llm="gpt-4o",
        verbose=True,
        allow_delegation=False,
        tools=[stagehand_tool],
    )

    # Define a research task that demonstrates all three primitives
    research_task = Task(
        description=(
            "Demonstrate Stagehand capabilities by performing the following steps:\n"
            "1. Go to https://www.stagehand.dev\n"
            "2. Extract all the text content from the page\n"
            "3. Find the Docs link and click on it\n"
            "4. Go to https://httpbin.org/forms/post and observe what elements are available on the page\n"
            "5. Provide a summary of what you learned about using these different commands"
        ),
        expected_output=(
            "A demonstration of all three Stagehand primitives (act, extract, observe) "
            "with examples of how each was used and what information was gathered."
        ),
        agent=researcher,
    )

    # Alternative task: Real research using the primitives
    web_research_task = Task(
        description=(
            "Go to google.com and search for 'Stagehand'.\n"
            "Then extract the first search result."
        ),
        expected_output=(
            "A summary report about Stagehand's capabilities and pricing, demonstrating how "
            "the different primitives can be used together for effective web research."
        ),
        agent=researcher,
    )

    # Set up the crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],  # You can switch this to web_research_task if you prefer
        verbose=True,
        process=Process.sequential,
    )

    # Run the crew and get the result
    result = crew.kickoff()

    print("\n==== RESULTS ====\n")
    print(result)

# Resources are automatically cleaned up when exiting the context manager
