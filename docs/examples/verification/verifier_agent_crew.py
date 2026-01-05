from crewai import Agent, Task, Crew, Process

# -----------------------------
# Agents
# -----------------------------
generator_agent = Agent(
    role="Content Generator",
    goal="Generate a clear and accurate answer to the given question",
    backstory="Expert assistant generating initial responses",
    verbose=True
)

verifier_agent = Agent(
    role="Verifier Agent",
    goal="Verify factual accuracy and detect hallucinations",
    backstory="Critical evaluator of AI outputs",
    verbose=True
)

# -----------------------------
# Tasks
# -----------------------------
generate_task = Task(
    description="Explain the primary causes of climate change in simple terms.",
    expected_output="A factually correct explanation of climate change causes.",
    agent=generator_agent
)

verify_task = Task(
    description=(
        "Review the previous response for factual accuracy and hallucinations.\n"
        "Return JSON with verdict, confidence, issues, suggested_fix."
    ),
    expected_output="A JSON verification report.",
    agent=verifier_agent
)

# -----------------------------
# Crew
# -----------------------------
crew = Crew(
    agents=[generator_agent, verifier_agent],
    tasks=[generate_task, verify_task],
    process=Process.sequential
)

if __name__ == "__main__":
    print(crew.kickoff())
