# run_crew.py
# Minimal CrewAI setup using:
# - Planner: OpenAI o3 (reasoning_effort='low' for cheaper "flex" mode)
# - Sub-agents: Vertex AI Gemini 2.0 Flash-Lite via service account
#
# Prereqs (PowerShell examples):
#   pip install crewai google-cloud-aiplatform
#   $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"
#   $env:VERTEXAI_PROJECT = "<gcp-project-id>"
#   $env:VERTEXAI_LOCATION = "us-central1"  # or your region
#   $env:OPENAI_API_KEY = "<your-openai-key>"

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

# Vertex-backed Gemini (Flash-Lite). If you prefer the direct Gemini API, use
#   GEMINI_MODEL = "gemini/gemini-2.0-flash-lite-001"
GEMINI_MODEL = "vertex_ai/gemini-2.0-flash-lite-001"

# Planner LLM (OpenAI o3) – low reasoning effort = flex/cheapest
planner_llm = LLM(
    model="o3",
    reasoning_effort="low",
)

# --- Agents (Gemini via Vertex AI) ---
architect = Agent(
    role="Crew Architect",
    goal="Draft a clear, modular crew design that can create and improve other crews.",
    backstory=(
        "Expert in decomposing problems into specialized agents, tasks, and feedback loops."
    ),
    llm=GEMINI_MODEL,
    verbose=True,
)

implementer = Agent(
    role="Crew Implementer",
    goal=(
        "Generate minimal, runnable crew specs/code with tests and iteration hooks."
    ),
    backstory=(
        "Focuses on small, verifiable increments and tooling for rapid iteration."
    ),
    llm=GEMINI_MODEL,
    verbose=True,
)

evaluator = Agent(
    role="Crew Evaluator",
    goal=(
        "Propose tests and quick evaluations to refine the generated crew."
    ),
    backstory=(
        "Ensures each iteration has measurable outcomes and cheaper test loops."
    ),
    llm=GEMINI_MODEL,
    verbose=True,
)

# --- Tasks ---
design_task = Task(
    description=(
        "Produce a concise spec for a new crew that can generate and refine other crews. "
        "Include roles, tools, and a simple iteration process (design -> implement -> test -> refine). "
        "Target low cost and fast validation cycles."
    ),
    agent=architect,
)

implement_task = Task(
    description=(
        "Take the spec and output a minimal runnable skeleton (pseudo or code outline) "
        "that a developer can paste into a project. Keep dependencies minimal."
    ),
    agent=implementer,
)

evaluate_task = Task(
    description=(
        "Write a brief evaluation checklist and a tiny test plan to validate the skeleton. "
        "Recommend next iteration steps."
    ),
    agent=evaluator,
)

# --- Crew ---
crew = Crew(
    agents=[architect, implementer, evaluator],
    tasks=[design_task, implement_task, evaluate_task],
    process=Process.sequential,
    planning=True,            # enable plan-before-execute
    planning_llm=planner_llm, # OpenAI o3 as the planner
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== Final Output ===\n")
    print(result)