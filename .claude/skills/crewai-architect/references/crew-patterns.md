# Crew Patterns Reference

## Table of Contents
- [YAML Configuration](#yaml-configuration)
- [Crew Class Structure](#crew-class-structure)
- [Process Types](#process-types)
- [Task Dependencies](#task-dependencies)
- [Agent Delegation](#agent-delegation)
- [Memory and Knowledge](#memory-and-knowledge)

## YAML Configuration

### Creating a Crew with CLI

```bash
crewai flow add-crew research_crew
```

Creates:
```
src/project/crews/research_crew/
├── __init__.py
├── research_crew.py
└── config/
    ├── agents.yaml
    └── tasks.yaml
```

### agents.yaml

```yaml
researcher:
  role: "Senior Research Analyst"
  goal: "Conduct thorough research on {topic} and identify key insights"
  backstory: |
    You are a veteran researcher with 15 years of experience
    in market analysis. You excel at finding hidden patterns
    and connecting disparate data points.
  verbose: true
  allow_delegation: false

analyst:
  role: "Data Analyst"
  goal: "Analyze research data and produce actionable recommendations"
  backstory: |
    Expert at transforming raw research into strategic insights.
    Known for clear, data-driven conclusions.
  verbose: true
  allow_delegation: true
```

**Variable interpolation:** Use `{variable}` for dynamic values passed at kickoff.

### tasks.yaml

```yaml
research_task:
  description: |
    Research the following topic thoroughly: {topic}

    Focus on:
    - Current market trends
    - Key players and competitors
    - Recent developments (last 6 months)
    - Potential risks and opportunities
  expected_output: |
    Comprehensive research report with:
    - Executive summary
    - Detailed findings
    - Data sources cited
  agent: researcher

analysis_task:
  description: |
    Analyze the research findings and provide recommendations.
    Consider: {analysis_focus}
  expected_output: |
    Strategic analysis with:
    - Key insights
    - Recommendations (prioritized)
    - Risk assessment
  agent: analyst
  context:
    - research_task
```

## Crew Class Structure

```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class ResearchCrew:
    """Research crew for thorough topic investigation."""

    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            tools=[SearchTool(), WebScrapeTool()],
            verbose=True,
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],  # type: ignore[index]
            tools=[AnalysisTool()],
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"],  # type: ignore[index]
            output_pydantic=AnalysisReport,  # Structured output
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

**Critical:** Always use `# type: ignore[index]` for config access.

## Process Types

### Sequential (Default)

Tasks execute in order, each receiving context from previous.

```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
    )
```

### Hierarchical

Manager agent coordinates work distribution.

```python
from crewai import LLM

@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,
        manager_llm=LLM(model="gpt-4o"),  # Required for hierarchical
    )
```

### Parallel Task Groups

```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        parallel_task_execution=True,  # Tasks without dependencies run parallel
    )
```

## Task Dependencies

### Context Chaining

```yaml
# tasks.yaml
gather_task:
  description: "Gather initial data"
  agent: gatherer

analyze_task:
  description: "Analyze gathered data"
  agent: analyst
  context:
    - gather_task  # Receives gather_task output

report_task:
  description: "Write final report"
  agent: writer
  context:
    - gather_task    # Has access to both
    - analyze_task   # previous outputs
```

### Programmatic Context

```python
@task
def synthesis_task(self) -> Task:
    return Task(
        config=self.tasks_config["synthesis_task"],  # type: ignore[index]
        context=[self.research_task(), self.analysis_task()],
    )
```

## Agent Delegation

### Enable Delegation

```yaml
# agents.yaml
manager:
  role: "Project Manager"
  goal: "Coordinate team and delegate effectively"
  allow_delegation: true  # Can delegate to other agents

specialist:
  role: "Technical Specialist"
  goal: "Handle technical implementation"
  allow_delegation: false  # Handles own work
```

### Delegation in Hierarchical

```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,
        manager_llm=LLM(model="gpt-4o"),
        manager_agent=self.manager(),  # Optional custom manager
    )
```

## Memory and Knowledge

### Enable Memory

```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        memory=True,  # Enable short-term memory
        verbose=True,
    )
```

### Knowledge Sources

```python
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

@crew
def crew(self) -> Crew:
    knowledge = TextFileKnowledgeSource(
        file_paths=["docs/guidelines.md", "docs/reference.md"]
    )
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        knowledge_sources=[knowledge],
    )
```

## Output Patterns

### Structured Task Output

```python
from pydantic import BaseModel, Field

class ResearchReport(BaseModel):
    summary: str
    findings: list[str]
    sources: list[str]
    confidence: float = Field(ge=0, le=1)

@task
def research_task(self) -> Task:
    return Task(
        config=self.tasks_config["research_task"],  # type: ignore[index]
        output_pydantic=ResearchReport,
    )
```

### File Output

```python
@task
def report_task(self) -> Task:
    return Task(
        config=self.tasks_config["report_task"],  # type: ignore[index]
        output_file="outputs/report.md",
    )
```

### Accessing Outputs

```python
# In flow method
result = ResearchCrew().crew().kickoff(inputs={"topic": "AI"})

# Raw output
print(result.raw)

# Structured output (if output_pydantic set)
report = result.pydantic
print(report.summary)

# Dictionary access
print(result["summary"])

# JSON (if output_json set)
data = result.json
```

## Using Crew in Flow

```python
from project_name.crews.research_crew.research_crew import ResearchCrew

class ResearchFlow(Flow[State]):
    @listen(validate_input)
    def do_research(self, validated):
        crew = ResearchCrew().crew()
        result = crew.kickoff(inputs={
            "topic": self.state.topic,
            "analysis_focus": self.state.focus
        })
        self.state.research = result.pydantic
        return result
```
