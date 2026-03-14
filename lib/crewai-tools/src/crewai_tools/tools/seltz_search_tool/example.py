"""Example usage of the SeltzSearchTool with a CrewAI agent.

Requires:
    pip install crewai crewai-tools seltz

Setup:
    cp .env.example .env
    # Edit .env with your real API keys
"""

import logging

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from crewai_tools import SeltzSearchTool


load_dotenv()

logger = logging.getLogger(__name__)

# Initialize with defaults (reads SELTZ_API_KEY from environment)
seltz_tool = SeltzSearchTool()

# Or customize the search behavior
# seltz_tool = SeltzSearchTool(
#     max_documents=10,
#     context="Focus on recent AI research papers and announcements",
#     profile="research",
# )

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, source-backed information on any topic",
    backstory="An expert researcher who synthesizes web knowledge into clear summaries.",
    tools=[seltz_tool],
    verbose=True,
)

research_task = Task(
    description="Research the latest developments in AI agents and multi-agent systems. "
    "Focus on practical applications and recent breakthroughs.",
    expected_output="A concise summary of the top developments, with source URLs.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[research_task])
result = crew.kickoff()
logger.info(result)
