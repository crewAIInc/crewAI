### Example 1: An EU economic analyst agent backed by real Eurostat data

This example builds a simple CrewAI workflow where an agent answers questions
about EU statistics using the `EurostatTool`, instead of relying on the
model's own (possibly stale or invented) knowledge.

```python
from crewai import Agent, Crew, Process, Task
from crewai_tools import EurostatTool

tool = EurostatTool()

analyst = Agent(
    role="EU Economic Analyst",
    goal="Answer questions about EU statistics using real, current Eurostat data",
    backstory=(
        "An analyst who always cites real published figures and their time "
        "period instead of guessing from memory."
    ),
    tools=[tool],
    verbose=True,
)

task = Task(
    description=(
        "What is Germany's most recent monthly unemployment rate, and how "
        "does it compare to France's over the same period?"
    ),
    expected_output=(
        "The unemployment rate for Germany and France for the same recent "
        "month, each with its time period, sourced via the Eurostat tool."
    ),
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=True)

result = crew.kickoff()
print(result.raw)
```
