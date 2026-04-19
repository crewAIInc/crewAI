"""
Example: guard an AI agent's input with DeepKeep.

Prerequisites
-------------
pip install 'crewai[tools]'

Set your credentials before running:
    export DEEPKEEP_SUBDOMAIN="acme"
    export DEEPKEEP_API_KEY="dk_..."
    export OPENAI_API_KEY="sk_..."
"""

import os

from crewai import Agent, Crew, Task

from crewai_tools.tools.deepkeep_tool import (
    DeepKeepCheckInputTool,
    DeepKeepCreateConversationTool,
)

SUBDOMAIN = os.environ["DEEPKEEP_SUBDOMAIN"]
API_KEY = os.environ["DEEPKEEP_API_KEY"]
FIREWALL_ID = "your-firewall-id-here"

create_conversation_tool = DeepKeepCreateConversationTool(
    subdomain=SUBDOMAIN,
    api_key=API_KEY,
)
check_input_tool = DeepKeepCheckInputTool(
    subdomain=SUBDOMAIN,
    api_key=API_KEY,
)

guard_agent = Agent(
    role="AI Security Guard",
    goal="Ensure all user inputs comply with company AI policies before they reach other agents.",
    backstory=(
        "You are a security specialist responsible for enforcing AI guardrails. "
        "You use the DeepKeep firewall to screen every user message."
    ),
    tools=[create_conversation_tool, check_input_tool],
    verbose=True,
)

check_task = Task(
    description=(
        f"1. Create a new conversation in firewall '{FIREWALL_ID}'.\n"
        f"2. Check the following user input:\n\n"
        f"   'How do I reset my password?'\n\n"
        f"3. Report whether the input was flagged, and include the violation details if any."
    ),
    expected_output="A summary of the firewall check result, including any violations found.",
    agent=guard_agent,
)

crew = Crew(agents=[guard_agent], tasks=[check_task])
result = crew.kickoff()
print(result)
