# ==============================================================================
# Creating a LangGraph agent
# ==============================================================================

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

model = ChatOpenAI(
    base_url="llama70b.wayflow.oraclecorp.com",
    model="/storage/models/Llama-3.1-70B-Instruct",
    api_key=SecretStr("t"),
)

def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


assistant = create_react_agent(
    name="langgraph_assistant",
    model=model,
    tools=[add, multiply, divide],
    prompt="Use tools to solve tasks.",
)

print("Created the LangGraph agent")

# ==============================================================================
# Exporting the WayFlow agent to AgentSpec
# ==============================================================================

import json
from langgraph_agentspec_adapter import AgentSpecExporter

serialized_assistant = AgentSpecExporter().to_json(assistant)

json.dump(json.loads(serialized_assistant), open('agentspec_langgraph_agent.json', 'w'), indent=4)

print("Exported the LangGraph agent to AgentSpec")

# ==============================================================================
# Loading the AgentSpec agent in CrewAI via AgentSpecAgentAdapter
# ==============================================================================

from crewai import Crew, Task
from crewai.agents.agent_adapters.agentspec.agentspec_adapter import AgentSpecAgentAdapter

tool_registry = {"add": add, "multiply": multiply, "divide": divide}

with open('agentspec_langgraph_agent.json', 'r') as f:
    agentspec_json = f.read()

crewai_assistant = AgentSpecAgentAdapter(
    agentspec_agent_json=agentspec_json,
    tool_registry=tool_registry
)

print("Loaded AgentSpec into CrewAI using AgentSpecAgentAdapter")

# ==============================================================================
# Running the agent inside of a Crew
# ==============================================================================

task = Task(
    description="{user_input}",
    expected_output="A helpful, concise reply to the user.",
    agent=crewai_assistant,
)
crew = Crew(agents=[crewai_assistant], tasks=[task])

while True:
    user_input = input("USER  >>> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = crew.kickoff(inputs={"user_input": user_input})
    print("AGENT >>>", response)
