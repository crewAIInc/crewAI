from crewai import Agent, Crew, Task

from crewai_tools import ScreenContextAgentTool


# ScreenContextAgent must be installed, running/configured, and this client approved.
screen_context_tool = ScreenContextAgentTool(
    command="screen-context",
    command_args=["serve", "--profile", "standard", "--transport", "stdio"],
    env={"SCREEN_CONTEXT_CLIENT_TOKEN": "<client-token>"},
)

researcher = Agent(
    role="Research assistant",
    goal="Answer questions using relevant information the user explicitly asks you to find",
    backstory="You retrieve prior screen history only when the user asks about it.",
    tools=[screen_context_tool],
)

recent_context = Task(
    description=(
        "The user explicitly asks: Find the error message I saw in my browser "
        "in the last 30 minutes. Call ScreenContextAgent history search with "
        "query='error', since_minutes=30, and limit=5, then report the app "
        "and timestamp for matching text."
    ),
    expected_output="Matching screen text with its source app and timestamp, or no matches.",
    agent=researcher,
)

Crew(agents=[researcher], tasks=[recent_context]).kickoff()
