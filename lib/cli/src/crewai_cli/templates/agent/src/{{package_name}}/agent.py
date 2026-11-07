import os
from pathlib import Path

from crewai import LLM, Agent

from .tools.lookup_plan import LookupPlanTool
from .tools.support_checklist import SupportChecklistTool
from .tools.support_hours import SupportHoursTool


agent = Agent(
    conversational=True,
    role="General-purpose AI agent",
    goal="Help users accomplish tasks accurately using the available tools and skills.",
    backstory=(
        "# Identity\n"
        "You are a general-purpose AI agent powered by CrewAI.\n\n"
        "# Customization\n"
        "Your behavior and capabilities are defined by this project's agent definition, "
        "tools, and skills. Customize these files for the user's needs. "
        "The included support tools are examples; use them only when relevant. "
        "Be clear about the capabilities available in this project and do not claim "
        "external actions without a tool result."
    ),
    llm=LLM(model=os.getenv("MODEL", "openai/gpt-6.1-sol"), api="responses"),
    tools=[SupportHoursTool(), LookupPlanTool(), SupportChecklistTool()],
    skills=[Path(__file__).parent / "skills"],
    max_iter=8,
)
