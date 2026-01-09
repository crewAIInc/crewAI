---
name: crewai-architect
description: |
  Guide for architecting CrewAI applications with mastery of different execution patterns.
  Use this skill when: (1) Designing new CrewAI projects, (2) Choosing between direct LLM calls vs single agents vs crews within flows,
  (3) Implementing flow-based orchestration with @start/@listen/@router decorators, (4) Embedding single agents or LLM calls within flow methods,
  (5) Adding crews to flows with `crewai flow add-crew`, (6) Choosing kickoff methods (kickoff, kickoff_async, akickoff),
  (7) Implementing structured outputs with Pydantic. Always start with Flows and add agency incrementally.
---

# CrewAI Architecture Guide

**Core Principle: Start with Flows, add agency as needed.**

Flows provide deterministic orchestration. Within flow methods, add agency incrementally:
1. Direct LLM calls (simplest)
2. Single agents (when tools needed)
3. Crews (when collaboration needed)

## Architecture Decision Framework

**Always start with a Flow.** Then choose the right level of agency for each step:

| Agency Level | Within Flow Method | When to Use |
|--------------|-------------------|-------------|
| **Direct LLM** | `llm.call(messages)` | Structured extraction, no tools |
| **Single Agent** | `Agent(...).kickoff()` | Tool usage, single perspective |
| **Crew** | `MyCrew().crew().kickoff()` | Multi-agent collaboration |

### Decision Tree for Each Flow Step

```
What does this step need?
├── Simple structured response → Direct LLM call
├── Tool usage, single perspective → Single Agent (inline)
└── Multi-perspective reasoning → Crew (use add-crew command)
```

## Multi-Agent Pattern Selection

For complex applications requiring multiple agents working together, choose the appropriate orchestration pattern:

| Pattern | Best For | Key Construct |
|---------|----------|---------------|
| **Generator-Critic** | Quality assurance with validation loops | Flow + @router + "revise" loop |
| **Iterative Refinement** | Progressive improvement to threshold | Flow + iteration counter |
| **Orchestrator-Worker** | Dynamic task decomposition | Flow + dynamic Agent() + asyncio.gather |
| **Task Guardrails** | Output validation with auto-retry | Task(guardrail=func, guardrail_max_retries=N) |
| **State Persistence** | Crash recovery, HITL resume | @persist() on Flow class |
| **HITL Webhooks** | Enterprise approval workflows | humanInputWebhook + /resume API |
| **Custom Manager** | Coordinated hierarchical delegation | Crew(manager_agent=..., process=hierarchical) |
| **Composite** | Enterprise production systems | Multiple patterns combined |

### Quick Pattern Selection

```
Output quality critical? → Generator-Critic or Task Guardrails
Iterative improvement needed? → Iterative Refinement
Unknown task complexity? → Orchestrator-Worker (Dynamic Spawning)
Independent parallel tasks? → Parallel Fan-Out with asyncio.gather
Crash recovery needed? → State Persistence with @persist()
Human approval required? → HITL Webhooks
Enterprise production? → Composite (combine patterns)
```

See [references/multi-agent-patterns.md](references/multi-agent-patterns.md) for complete implementations.

---

## Pattern 1: Flow with Direct LLM Calls

**Use when:** Structured extraction, classification, simple transformations.

```python
from crewai.flow.flow import Flow, start, listen
from crewai import LLM
from pydantic import BaseModel

class TaskClassification(BaseModel):
    category: str
    priority: int
    confidence: float

class PipelineState(BaseModel):
    input: str = ""
    classification: TaskClassification | None = None

class ClassificationFlow(Flow[PipelineState]):
    def __init__(self):
        super().__init__()
        self.llm = LLM(model="gpt-4o")

    @start()
    def classify_input(self):
        llm = LLM(model="gpt-4o", response_format=TaskClassification)
        result = llm.call(messages=[
            {"role": "user", "content": f"Classify: {self.state.input}"}
        ])
        self.state.classification = result  # Returns typed model directly
        return result

# Execute
flow = ClassificationFlow()
flow.state.input = "Urgent bug in production"
result = flow.kickoff()
```

## Pattern 2: Flow with Single Agents

**Use when:** Step requires tools, memory, or multi-step reasoning from one perspective.

**Define agents directly in flow methods:**

```python
from crewai.flow.flow import Flow, start, listen, router
from crewai import Agent, LLM
from pydantic import BaseModel

class AnalysisState(BaseModel):
    data: str = ""
    analysis: str = ""
    needs_deep_dive: bool = False

class AnalysisFlow(Flow[AnalysisState]):

    @start()
    def quick_scan(self):
        # Single agent defined inline for tool usage
        scanner = Agent(
            role="Data Scanner",
            goal="Quickly scan data for anomalies",
            backstory="Expert at rapid data assessment",
            tools=[DataScanTool()],
            llm=LLM(model="gpt-4o")
        )
        result = scanner.kickoff(f"Scan: {self.state.data}")
        self.state.needs_deep_dive = "anomaly" in result.raw.lower()
        return result

    @router(quick_scan)
    def route_analysis(self):
        return "deep_dive" if self.state.needs_deep_dive else "summary"

    @listen("deep_dive")
    def detailed_analysis(self):
        # Another inline agent for different task
        analyst = Agent(
            role="Deep Analyst",
            goal="Conduct thorough analysis",
            backstory="Meticulous investigator",
            tools=[AnalysisTool(), ChartTool()]
        )
        result = analyst.kickoff(f"Deep dive: {self.state.data}")
        self.state.analysis = result.raw
        return result

    @listen("summary")
    def quick_summary(self):
        # Direct LLM call when no tools needed
        llm = LLM(model="gpt-4o")
        result = llm.call([{"role": "user", "content": f"Summarize: {self.state.data}"}])
        self.state.analysis = result
        return result
```

**Async agent execution:**
```python
@listen(some_method)
async def async_analysis(self, data):
    agent = Agent(role="Analyst", ...)
    result = await agent.kickoff_async(f"Analyze: {data}")
    return result
```

## Pattern 3: Flow with Crews

**Use when:** Step requires multi-agent collaboration and autonomous problem-solving.

### Adding a Crew to a Flow

Use the CLI command:
```bash
crewai flow add-crew research_crew
```

This creates the crew structure under `src/your_project/crews/research_crew/`:
```
research_crew/
├── __init__.py
├── research_crew.py
└── config/
    ├── agents.yaml
    └── tasks.yaml
```

### Using Crews in Flow Methods

```python
from crewai.flow.flow import Flow, start, listen, router, or_
from pydantic import BaseModel
from .crews.research_crew.research_crew import ResearchCrew
from .crews.writing_crew.writing_crew import WritingCrew

class ContentState(BaseModel):
    topic: str = ""
    research: str = ""
    article: str = ""
    confidence: float = 0.0

class ContentPipeline(Flow[ContentState]):

    @start()
    def validate_topic(self):
        # Quick LLM validation
        llm = LLM(model="gpt-4o", response_format=TopicValidation)
        return llm.call([{"role": "user", "content": f"Validate topic: {self.state.topic}"}])

    @listen(validate_topic)
    def research_topic(self, validation):
        # Crew for complex research
        crew = ResearchCrew().crew()
        result = crew.kickoff(inputs={"topic": self.state.topic})
        self.state.research = result.raw
        self.state.confidence = 0.85  # From crew output
        return result

    @router(research_topic)
    def route_by_confidence(self):
        if self.state.confidence > 0.8:
            return "write_article"
        return "needs_more_research"

    @listen("write_article")
    def write_content(self):
        # Another crew for writing
        crew = WritingCrew().crew()
        result = crew.kickoff(inputs={
            "topic": self.state.topic,
            "research": self.state.research
        })
        self.state.article = result.raw
        return result

    @listen("needs_more_research")
    def request_human_input(self):
        return "Research inconclusive - human review needed"
```

## Flow Decorators Reference

| Decorator | Purpose | Example |
|-----------|---------|---------|
| `@start()` | Entry point (multiple allowed, run parallel) | `@start()` |
| `@listen(method)` | Trigger on method completion | `@listen(validate)` |
| `@listen("label")` | Trigger on router label | `@listen("approved")` |
| `@router(method)` | Conditional routing, returns label | Returns `"approved"` or `"rejected"` |
| `and_(a, b)` | Trigger when ALL complete | `@listen(and_(task_a, task_b))` |
| `or_(a, b)` | Trigger when ANY completes | `@listen(or_("pass", "fail"))` |

### Parallel Starts

```python
class ParallelFlow(Flow[State]):
    @start()  # Runs in parallel
    def fetch_data_a(self):
        return "Data A"

    @start()  # Runs in parallel
    def fetch_data_b(self):
        return "Data B"

    @listen(and_(fetch_data_a, fetch_data_b))
    def combine_results(self, result_a, result_b):
        return f"{result_a} + {result_b}"
```

### Conditional Routing

```python
@router(process_step)
def decide_path(self):
    if self.state.score > 90:
        return "excellent"
    elif self.state.score > 70:
        return "good"
    return "needs_improvement"

@listen("excellent")
def handle_excellent(self): ...

@listen(or_("good", "needs_improvement"))
def handle_other(self): ...
```

## Kickoff Methods Reference

### Crew Kickoff (within Flow methods)

| Method | Use Case |
|--------|----------|
| `crew.kickoff(inputs={})` | Standard synchronous |
| `await crew.kickoff_async(inputs={})` | Thread-based async |
| `await crew.akickoff(inputs={})` | Native async (preferred) |
| `crew.kickoff_for_each(inputs=[...])` | Sequential batch |
| `await crew.akickoff_for_each([...])` | Concurrent batch (preferred) |

### Flow Kickoff

```python
flow = MyFlow()
flow.state.input = "data"  # Set state
result = flow.kickoff()
print(flow.state.result)  # Access final state
```

## Structured Output Patterns

### At LLM Level (Direct calls)

```python
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    summary: str
    key_points: list[str]
    confidence: float = Field(ge=0, le=1)

@start()
def analyze(self):
    llm = LLM(model="gpt-4o", response_format=Analysis)
    result = llm.call([...])
    return result  # Returns typed Analysis instance directly
```

### At Agent Level

```python
@listen(previous_step)
def agent_analysis(self, data):
    agent = Agent(role="Analyst", ...)
    result = agent.kickoff(
        f"Analyze: {data}",
        response_format=Analysis
    )
    return result.pydantic
```

### At Task Level (Crew)

```python
# In crew task definition
task = Task(
    description="Generate report",
    expected_output="Structured report",
    agent=analyst,
    output_pydantic=Report  # Enforces structure
)

# Access in flow
result = crew.kickoff(inputs={...})
report = result.pydantic  # Typed Report
```

## Complete Example: Adaptive Research Pipeline

```python
from crewai.flow.flow import Flow, start, listen, router, or_
from crewai import Agent, LLM
from pydantic import BaseModel
from .crews.research_crew.research_crew import ResearchCrew

class ResearchState(BaseModel):
    query: str = ""
    complexity: str = "simple"
    findings: str = ""
    confidence: float = 0.0

class AdaptiveResearch(Flow[ResearchState]):

    @start()
    def classify_query(self):
        """LLM call for quick classification"""
        llm = LLM(model="gpt-4o", response_format=QueryClassification)
        result = llm.call([
            {"role": "user", "content": f"Classify complexity: {self.state.query}"}
        ])
        self.state.complexity = result.complexity  # LLM returns model directly
        return result

    @router(classify_query)
    def route_by_complexity(self):
        return self.state.complexity  # "simple", "moderate", or "complex"

    @listen("simple")
    def quick_search(self):
        """Single agent for simple queries"""
        searcher = Agent(
            role="Quick Researcher",
            goal="Find answer efficiently",
            tools=[SearchTool()],
            llm=LLM(model="gpt-4o-mini")  # Faster model
        )
        result = searcher.kickoff(self.state.query)
        self.state.findings = result.raw
        self.state.confidence = 0.7
        return result

    @listen("moderate")
    def standard_research(self):
        """Single agent with more tools"""
        researcher = Agent(
            role="Researcher",
            goal="Thorough research",
            tools=[SearchTool(), AnalysisTool()],
            llm=LLM(model="gpt-4o")
        )
        result = researcher.kickoff(self.state.query)
        self.state.findings = result.raw
        self.state.confidence = 0.85
        return result

    @listen("complex")
    def deep_research(self):
        """Full crew for complex queries"""
        crew = ResearchCrew().crew()
        result = crew.kickoff(inputs={"query": self.state.query})
        self.state.findings = result.raw
        self.state.confidence = 0.95
        return result

    @listen(or_("simple", "moderate", "complex"))
    def finalize(self):
        return {
            "findings": self.state.findings,
            "confidence": self.state.confidence
        }

# Execute
flow = AdaptiveResearch()
flow.state.query = "What are the implications of quantum computing on cryptography?"
result = flow.kickoff()
```

## Best Practices

1. **Start with Flow** - Always use Flow as the orchestration layer
2. **Add agency incrementally** - LLM → Agent → Crew, only as needed
3. **Use `crewai flow add-crew`** - For creating crews within flows
4. **Define agents inline** - In flow methods for single-use agents
5. **Initialize LLMs in `__init__`** - Reuse LLM instances for efficiency
6. **Type your state** - Always use Pydantic BaseModel
7. **Use structured outputs** - `response_format` and `output_pydantic`
8. **Prefer native async** - `akickoff()` over `kickoff_async()`
9. **Route by state** - Use `@router` for conditional paths

## Reference Files

- [references/multi-agent-patterns.md](references/multi-agent-patterns.md) - **Multi-agent orchestration**: Generator-Critic, Iterative Refinement, Orchestrator-Worker, Task Guardrails, State Persistence, HITL Webhooks, Composite patterns
- [references/flow-patterns.md](references/flow-patterns.md) - Advanced flow patterns, HITL, resumable flows
- [references/crew-patterns.md](references/crew-patterns.md) - YAML config, process types, delegation
- [references/llm-patterns.md](references/llm-patterns.md) - Custom LLM integration, providers
