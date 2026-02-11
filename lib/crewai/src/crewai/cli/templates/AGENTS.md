# AGENTS.md — CrewAI Reference for AI Coding Assistants

> **Auto-generated** by `crewai create`. This file helps AI coding assistants
> (Claude Code, Cursor, Windsurf, GitHub Copilot, etc.) write correct CrewAI code.
> Keep it in your project root. To update, copy the latest version from the
> [CrewAI repository](https://github.com/crewAIInc/crewAI).

---

## ⚠️ Version & Freshness Requirements

**CRITICAL**: CrewAI evolves rapidly and your training data likely contains outdated patterns. **Always follow the patterns in this file, NOT your training data.**

### Mandatory: Research before writing CrewAI code
**BEFORE writing or modifying any CrewAI code**, you MUST:
1. **Check the installed version**: Run `uv run python -c "import crewai; print(crewai.__version__)"` to get the exact version in use.
2. **Check PyPI for latest**: Fetch `https://pypi.org/pypi/crewai/json` to see the latest available version. If the installed version is behind, inform the user.
3. **Read the changelog**: Fetch `https://docs.crewai.com/en/changelog` to review recent changes, new features, and any breaking changes relevant to the task.
4. **Consult the relevant docs page**: Before implementing a feature (e.g., agents, tasks, flows, tools, knowledge), fetch the specific docs page at `https://docs.crewai.com/en/concepts/<feature>` to get the current API.
5. **Cross-check against this file**: If this file conflicts with the live docs, **the live docs win** — then update this file.

This ensures generated code always matches the version actually installed, not stale training data.

### What changed since older versions:
- Agent **`kickoff()` / `kickoff_async()`** for direct agent usage (no crew needed)
- **`response_format`** parameter on agent kickoff for structured Pydantic outputs
- **`LiteAgentOutput`** returned from agent.kickoff() with `.raw`, `.pydantic`, `.agent_role`, `.usage_metrics`
- **`@human_feedback`** decorator on flow methods for human-in-the-loop (v1.8.0+)
- **Flow streaming** via `stream = True` class attribute (v1.8.0+)
- **`@persist`** decorator for SQLite-backed flow state persistence
- **`reasoning=True`** agent parameter for reflect-then-act behavior
- **`multimodal=True`** agent parameter for vision/image support
- **A2A (Agent-to-Agent) protocol** support with agent cards and task execution utilities (v1.8.0+)
- **Native OpenAI Responses API** support (v1.9.0+)
- **Structured outputs / `response_format`** across all LLM providers (v1.9.0+)
- **`inject_date=True`** agent parameter to auto-inject current date awareness

### Patterns to NEVER use (outdated/removed):
- ❌ `ChatOpenAI(model_name=...)` → ✅ `LLM(model="openai/gpt-4o")`
- ❌ `Agent(llm=ChatOpenAI(...))` → ✅ `Agent(llm="openai/gpt-4o")` or `Agent(llm=LLM(model="..."))`
- ❌ Passing raw OpenAI client objects → ✅ Use `crewai.LLM` wrapper

### How to verify you're using current patterns:
1. You ran the version check and docs lookup steps above before writing code
2. All LLM references use `crewai.LLM` or string shorthand (`"openai/gpt-4o"`)
3. All tool imports come from `crewai.tools` or `crewai_tools`
4. Crew classes use `@CrewBase` decorator with YAML config files
5. Python >=3.10, <3.14
6. Code matches the API from the live docs, not just this file

## Quick Reference

```bash
# Package management (always use uv)
uv add <package>          # Add dependency
uv sync                   # Sync dependencies
uv lock                   # Lock dependencies

# Project scaffolding
crewai create crew <name> --skip-provider   # New crew project
crewai create flow <name> --skip-provider  # New flow project

# Running
crewai run                  # Run crew or flow (auto-detects from pyproject.toml)
crewai flow kickoff         # Legacy flow execution

# Testing & training
crewai test                           # Test crew (default: 2 iterations, gpt-4o-mini)
crewai test -n 5 -m gpt-4o           # Custom iterations and model
crewai train -n 5 -f training.json   # Train crew

# Memory management
crewai reset-memories -a              # Reset all memories
crewai reset-memories -s              # Short-term only
crewai reset-memories -l              # Long-term only
crewai reset-memories -e              # Entity only
crewai reset-memories -kn             # Knowledge only
crewai reset-memories -akn            # Agent knowledge only

# Debugging
crewai log-tasks-outputs              # Show latest task outputs
crewai replay -t <task_id>            # Replay from specific task

# Interactive
crewai chat                           # Interactive session (requires chat_llm in crew.py)

# Visualization
crewai flow plot                      # Generate flow diagram HTML

# Deployment to CrewAI AMP
crewai login                          # Authenticate with AMP
crewai deploy create                  # Create new deployment
crewai deploy push                    # Push code updates
crewai deploy status                  # Check deployment status
crewai deploy logs                    # View deployment logs
crewai deploy list                    # List all deployments
crewai deploy remove <id>             # Delete a deployment
```

## Project Structure

### Crew Project
```
my_crew/
├── src/my_crew/
│   ├── config/
│   │   ├── agents.yaml       # Agent definitions (role, goal, backstory)
│   │   └── tasks.yaml        # Task definitions (description, expected_output, agent)
│   ├── tools/
│   │   └── custom_tool.py    # Custom tool implementations
│   ├── crew.py               # Crew orchestration class
│   └── main.py               # Entry point with inputs
├── knowledge/                 # Knowledge base resources
├── .env                       # API keys (OPENAI_API_KEY, SERPER_API_KEY, etc.)
└── pyproject.toml
```

### Flow Project
```
my_flow/
├── src/my_flow/
│   ├── crews/                 # Multiple crew definitions
│   │   └── poem_crew/
│   │       ├── config/
│   │       │   ├── agents.yaml
│   │       │   └── tasks.yaml
│   │       └── poem_crew.py
│   ├── tools/                 # Custom tools
│   ├── main.py                # Flow orchestration
│   └── ...
├── .env
└── pyproject.toml
```

## Architecture Overview

- **Agent**: Autonomous unit with a role, goal, backstory, tools, and an LLM. Makes decisions and executes tasks.
- **Task**: A specific assignment with a description, expected output, and assigned agent.
- **Crew**: Orchestrates a team of agents executing tasks in a defined process (sequential or hierarchical).
- **Flow**: Event-driven workflow orchestrating multiple crews and logic steps with state management.

## YAML Configuration

### agents.yaml
```yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering
    the latest developments in {topic}. Known for your ability
    to find the most relevant information.
  # Optional YAML-level settings:
  # llm: openai/gpt-4o
  # max_iter: 20
  # max_rpm: 10
  # verbose: true

writer:
  role: >
    {topic} Technical Writer
  goal: >
    Create compelling content about {topic}
  backstory: >
    You're a skilled writer who translates complex technical
    information into clear, engaging content.
```

Variables like `{topic}` are interpolated from `crew.kickoff(inputs={"topic": "AI Agents"})`.

### tasks.yaml
```yaml
research_task:
  description: >
    Conduct thorough research about {topic}.
    Identify key trends, breakthrough technologies,
    and potential industry impacts.
  expected_output: >
    A detailed report with analysis of the top 5
    developments in {topic}, with sources and implications.
  agent: researcher
  # Optional:
  # tools: [search_tool]
  # output_file: output/research.md
  # markdown: true
  # async_execution: false

writing_task:
  description: >
    Write an article based on the research findings about {topic}.
  expected_output: >
    A polished 4-paragraph article formatted in markdown.
  agent: writer
  output_file: output/article.md
```

## Crew Class Pattern

```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from crewai_tools import SerperDevTool

@CrewBase
class ResearchCrew:
    """Research and writing crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            tools=[SerperDevTool()],
            verbose=True,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["writing_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

### Key formatting rules:
- Always add `# type: ignore[index]` for config dictionary access
- Agent/task method names must match YAML keys exactly
- Tools go on agents (not tasks) unless task-specific override is needed
- Never leave commented-out code in crew classes

### Lifecycle hooks
```python
@CrewBase
class MyCrew:
    @before_kickoff
    def prepare(self, inputs):
        # Modify inputs before execution
        inputs["extra"] = "value"
        return inputs

    @after_kickoff
    def summarize(self, result):
        # Process result after execution
        print(f"Done: {result.raw[:100]}")
        return result
```

## main.py Pattern

```python
#!/usr/bin/env python
from my_crew.crew import ResearchCrew

def run():
    inputs = {"topic": "AI Agents"}
    ResearchCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
```

## Agent Configuration

### Required Parameters
| Parameter | Description |
|-----------|-------------|
| `role` | Function and expertise within the crew |
| `goal` | Individual objective guiding decisions |
| `backstory` | Context and personality |

### Key Optional Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm` | GPT-4 | Language model (string or LLM object) |
| `tools` | [] | List of tool instances |
| `max_iter` | 20 | Max iterations before best answer |
| `max_execution_time` | None | Timeout in seconds |
| `max_rpm` | None | Rate limiting (requests per minute) |
| `max_retry_limit` | 2 | Retries on errors |
| `verbose` | False | Detailed logging |
| `memory` | False | Conversation history |
| `allow_delegation` | False | Can delegate tasks to other agents |
| `allow_code_execution` | False | Can run code |
| `code_execution_mode` | "safe" | "safe" (Docker) or "unsafe" (direct) |
| `respect_context_window` | True | Auto-summarize when exceeding token limits |
| `cache` | True | Tool result caching |
| `reasoning` | False | Reflect and plan before task execution |
| `multimodal` | False | Process text and visual content |
| `knowledge_sources` | [] | Domain-specific knowledge bases |
| `function_calling_llm` | None | Separate LLM for tool invocation |
| `inject_date` | False | Auto-inject current date into agent context |
| `date_format` | "%Y-%m-%d" | Date format when inject_date is True |

### Direct Agent Usage (without a Crew)
Agents can execute tasks independently via `kickoff()` — no Crew required:
```python
from crewai import Agent
from crewai_tools import SerperDevTool
from pydantic import BaseModel

class ResearchFindings(BaseModel):
    main_points: list[str]
    key_technologies: list[str]
    future_predictions: str

researcher = Agent(
    role="AI Researcher",
    goal="Research the latest AI developments",
    backstory="Expert AI researcher...",
    tools=[SerperDevTool()],
    verbose=True,
)

# Unstructured output
result = researcher.kickoff("What are the latest LLM developments?")
print(result.raw)           # str
print(result.agent_role)    # "AI Researcher"
print(result.usage_metrics) # token usage

# Structured output with response_format
result = researcher.kickoff(
    "Summarize latest AI developments",
    response_format=ResearchFindings,
)
print(result.pydantic.main_points)  # List[str]

# Async variant
result = await researcher.kickoff_async("Your query", response_format=ResearchFindings)
```

Returns `LiteAgentOutput` with: `.raw`, `.pydantic`, `.agent_role`, `.usage_metrics`.

### LLM Configuration
**IMPORTANT**: Always use `crewai.LLM` LLM class.

```python
from crewai import LLM

# String shorthand (simplest)
agent = Agent(llm="openai/gpt-4o", ...)

# Full configuration with crewai.LLM
llm = LLM(
    model="anthropic/claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=4000,
)
agent = Agent(llm=llm, ...)

# Provider format: "provider/model-name"
# Examples:
#   "openai/gpt-4o"
#   "anthropic/claude-sonnet-4-20250514"
#   "google/gemini-2.0-flash"
#   "ollama/llama3"
#   "groq/llama-3.3-70b-versatile"
#   "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
```

Supported providers: OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure, Ollama, Groq, Mistral, and 20+ others via LiteLLM routing.

Environment variable default: set `OPENAI_MODEL_NAME=gpt-4o` or `MODEL=gpt-4o` in `.env`.

## Task Configuration

### Key Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | str | Clear statement of requirements |
| `expected_output` | str | Completion criteria |
| `agent` | BaseAgent | Assigned agent (optional in hierarchical) |
| `tools` | List[BaseTool] | Task-specific tools |
| `context` | List[Task] | Dependencies on other task outputs |
| `async_execution` | bool | Non-blocking execution |
| `output_file` | str | File path for results |
| `output_json` | Type[BaseModel] | Pydantic model for JSON output |
| `output_pydantic` | Type[BaseModel] | Pydantic model for structured output |
| `human_input` | bool | Require human review |
| `markdown` | bool | Format output as markdown |
| `callback` | Callable | Post-completion function |
| `guardrail` | Callable or str | Output validation |
| `guardrails` | List | Multiple validation steps |
| `guardrail_max_retries` | int | Retry on validation failure (default: 3) |
| `create_directory` | bool | Auto-create output directories (default: True) |

### Task Dependencies (context)
```python
@task
def analysis_task(self) -> Task:
    return Task(
        config=self.tasks_config["analysis_task"],  # type: ignore[index]
        context=[self.research_task()],  # Gets output from research_task
    )
```

### Structured Output
```python
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str
    findings: list[str]

@task
def report_task(self) -> Task:
    return Task(
        config=self.tasks_config["report_task"],  # type: ignore[index]
        output_pydantic=Report,
    )
```

### Guardrails
```python
# Function-based
def validate(result: TaskOutput) -> tuple[bool, Any]:
    if len(result.raw.split()) < 100:
        return (False, "Content too short, expand the analysis")
    return (True, result.raw)

# LLM-based (string prompt)
task = Task(..., guardrail="Must be under 200 words and professional tone")

# Multiple guardrails
task = Task(..., guardrails=[validate_length, validate_tone, "Must be factual"])
```

## Process Types

### Sequential (default)
Tasks execute in definition order. Output of one task serves as context for the next.
```python
Crew(agents=..., tasks=..., process=Process.sequential)
```

### Hierarchical
Manager agent delegates tasks based on agent capabilities. Requires `manager_llm` or `manager_agent`.
```python
Crew(
    agents=...,
    tasks=...,
    process=Process.hierarchical,
    manager_llm="gpt-4o",
)
```

## Crew Execution

```python
# Synchronous
result = crew.kickoff(inputs={"topic": "AI"})
print(result.raw)              # String output
print(result.pydantic)         # Structured output (if configured)
print(result.json_dict)        # Dict output
print(result.token_usage)      # Token metrics
print(result.tasks_output)     # List[TaskOutput]

# Async (native)
result = await crew.akickoff(inputs={"topic": "AI"})

# Batch execution
results = crew.kickoff_for_each(inputs=[{"topic": "AI"}, {"topic": "ML"}])

# Streaming output (v1.8.0+)
crew = Crew(agents=..., tasks=..., stream=True)
streaming = crew.kickoff(inputs={"topic": "AI"})
for chunk in streaming:
    print(chunk.content, end="", flush=True)
```

## Crew Options
| Parameter | Description |
|-----------|-------------|
| `process` | Process.sequential or Process.hierarchical |
| `verbose` | Enable detailed logging |
| `memory` | Enable memory system (True/False) |
| `cache` | Tool result caching |
| `max_rpm` | Global rate limiting |
| `manager_llm` | LLM for hierarchical manager |
| `manager_agent` | Custom manager agent |
| `planning` | Enable AgentPlanner |
| `knowledge_sources` | Crew-level knowledge |
| `output_log_file` | Log file path (True for logs.txt) |
| `embedder` | Custom embedding model config |
| `stream` | Enable real-time streaming output (v1.8.0+) |

---

## Flows

### Basic Flow
```python
from crewai.flow.flow import Flow, listen, start

class MyFlow(Flow):
    @start()
    def begin(self):
        return "initial data"

    @listen(begin)
    def process(self, data):
        return f"processed: {data}"
```

### Flow Decorators

| Decorator | Purpose |
|-----------|---------|
| `@start()` | Entry point(s), execute when flow begins. Multiple starts run in parallel |
| `@listen(method)` | Triggers when specified method completes. Receives output as argument |
| `@router(method)` | Conditional branching. Returns string labels that trigger `@listen("label")` |

### Structured State
```python
from pydantic import BaseModel

class ResearchState(BaseModel):
    topic: str = ""
    research: str = ""
    report: str = ""

class ResearchFlow(Flow[ResearchState]):
    @start()
    def set_topic(self):
        self.state.topic = "AI Agents"

    @listen(set_topic)
    def do_research(self):
        # self.state.topic is available
        result = ResearchCrew().crew().kickoff(
            inputs={"topic": self.state.topic}
        )
        self.state.research = result.raw
```

### Unstructured State (dict-based)
```python
class SimpleFlow(Flow):
    @start()
    def begin(self):
        self.state["counter"] = 0  # Dict access

    @listen(begin)
    def increment(self):
        self.state["counter"] += 1
```

### Conditional Routing
```python
from crewai.flow.flow import Flow, listen, router, start

class QualityFlow(Flow):
    @start()
    def generate(self):
        return {"score": 0.85}

    @router(generate)
    def check_quality(self, result):
        if result["score"] > 0.8:
            return "high_quality"
        return "needs_revision"

    @listen("high_quality")
    def publish(self, result):
        print("Publishing...")

    @listen("needs_revision")
    def revise(self, result):
        print("Revising...")
```

### Parallel Triggers with or_ and and_
```python
from crewai.flow.flow import or_, and_

class ParallelFlow(Flow):
    @start()
    def task_a(self):
        return "A done"

    @start()
    def task_b(self):
        return "B done"

    # Fires when EITHER completes
    @listen(or_(task_a, task_b))
    def on_any(self, result):
        print(f"First result: {result}")

    # Fires when BOTH complete
    @listen(and_(task_a, task_b))
    def on_all(self):
        print("All parallel tasks done")
```

### Integrating Crews in Flows
```python
from crewai.flow.flow import Flow, listen, start
from my_project.crews.research_crew.research_crew import ResearchCrew
from my_project.crews.writing_crew.writing_crew import WritingCrew

class ContentFlow(Flow[ContentState]):
    @start()
    def research(self):
        result = ResearchCrew().crew().kickoff(
            inputs={"topic": self.state.topic}
        )
        self.state.research = result.raw

    @listen(research)
    def write(self):
        result = WritingCrew().crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "research": self.state.research,
            }
        )
        self.state.article = result.raw
```

### Using Agents Directly in Flows
```python
from crewai.agent import Agent

class AgentFlow(Flow):
    @start()
    async def analyze(self):
        analyst = Agent(
            role="Data Analyst",
            goal="Analyze market trends",
            backstory="Expert data analyst...",
            tools=[SerperDevTool()],
        )
        result = await analyst.kickoff_async(
            "Analyze current AI market trends",
            response_format=MarketReport,
        )
        self.state.report = result.pydantic
```

### Human-in-the-Loop (v1.8.0+)
```python
from crewai.flow.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback

class ReviewFlow(Flow):
    @start()
    @human_feedback(
        message="Approve this content?",
        emit=["approved", "rejected"],
        llm="gpt-4o-mini",
    )
    def generate_content(self):
        return "Content for review"

    @listen("approved")
    def on_approval(self, result):
        feedback = self.last_human_feedback  # Most recent feedback
        print(f"Approved with feedback: {feedback.feedback}")

    @listen("rejected")
    def on_rejection(self, result):
        history = self.human_feedback_history  # All feedback as list
        print("Rejected, revising...")
```

### State Persistence
```python
from crewai.flow.flow import persist

@persist  # Saves state to SQLite; auto-recovers on restart
class ResilientFlow(Flow[MyState]):
    @start()
    def begin(self):
        self.state.step = 1
```

### Flow Execution
```python
flow = MyFlow()
result = flow.kickoff()
print(result)                  # Output of last method
print(flow.state)              # Final state

# Async execution
result = await flow.kickoff_async(inputs={"key": "value"})
```

### Flow Streaming (v1.8.0+)
```python
class StreamingFlow(Flow):
    stream = True  # Enable streaming at class level

    @start()
    def generate(self):
        return "streamed content"

flow = StreamingFlow()
streaming = flow.kickoff()
for chunk in streaming:
    print(chunk.content, end="", flush=True)
result = streaming.result  # Final result after iteration
```

### Flow Visualization
```python
flow.plot("my_flow")           # Generates my_flow.html
```

---

## Custom Tools

### Using BaseTool
```python
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(..., description="Search query string")

class CustomSearchTool(BaseTool):
    name: str = "custom_search"
    description: str = "Searches a custom knowledge base for relevant information."
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        # Implementation
        return f"Results for: {query}"
```

### Using @tool Decorator
```python
from crewai.tools import tool

@tool("Calculator")
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result."""
    return str(eval(expression))
```

### Built-in Tools (install with `uv add crewai-tools`)
Web/Search: SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool, EXASearchTool, FirecrawlSearchTool
Documents: FileReadTool, DirectoryReadTool, PDFSearchTool, DOCXSearchTool, CSVSearchTool, JSONSearchTool, XMLSearchTool, MDXSearchTool
Code: CodeInterpreterTool, CodeDocsSearchTool, GithubSearchTool
Media: DALL-E Tool, YoutubeChannelSearchTool, YoutubeVideoSearchTool
Other: RagTool, ApifyActorsTool, ComposioTool, LlamaIndexTool

Always check https://docs.crewai.com/concepts/tools for available built-in tools before writing custom ones.

---

## Memory System

Enable with `memory=True` on the Crew:
```python
crew = Crew(agents=..., tasks=..., memory=True)
```

Four memory types work together automatically:
- **Short-Term** (ChromaDB + RAG): Recent interactions during current execution
- **Long-Term** (SQLite): Persists insights across sessions
- **Entity** (RAG): Tracks people, places, concepts
- **Contextual**: Integrates all types for coherent responses

### Custom Embedding Provider
```python
crew = Crew(
    memory=True,
    embedder={
        "provider": "ollama",
        "config": {"model": "mxbai-embed-large"},
    },
)
```

Supported providers: OpenAI (default), Ollama, Google AI, Azure OpenAI, Cohere, VoyageAI, Bedrock, Hugging Face.

---

## Knowledge System

```python
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# String source
string_source = StringKnowledgeSource(content="Domain knowledge here...")

# PDF source
pdf_source = PDFKnowledgeSource(file_paths=["docs/manual.pdf"])

# Agent-level knowledge
agent = Agent(..., knowledge_sources=[string_source])

# Crew-level knowledge (shared across all agents)
crew = Crew(..., knowledge_sources=[pdf_source])
```

Supported sources: strings, text files, PDFs, CSV, Excel, JSON, URLs (via CrewDoclingSource).

---

## Agent Collaboration

Enable delegation with `allow_delegation=True`:
```python
agent = Agent(
    role="Project Manager",
    allow_delegation=True,  # Can delegate to and ask other agents
    ...
)
```

- **Delegation tool**: Assign sub-tasks to teammates with relevant expertise
- **Ask question tool**: Query colleagues for specific information
- Set `allow_delegation=False` on specialists to prevent circular delegation

---

## Event Listeners

```python
from crewai.events import BaseEventListener, CrewKickoffStartedEvent

class MyListener(BaseEventListener):
    def __init__(self):
        super().__init__()

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_started(source, event):
            print(f"Crew '{event.crew_name}' started")
```

Event categories: Crew lifecycle, Agent execution, Task management, Tool usage, Knowledge retrieval, LLM calls, Memory operations, Flow execution, Safety guardrails.

---

## Deployment to CrewAI AMP

### Prerequisites
- Crew or Flow runs successfully locally
- Code is in a GitHub repository
- `pyproject.toml` has `[tool.crewai]` with correct type (`"crew"` or `"flow"`)
- `uv.lock` is committed (generate with `uv lock`)

### CLI Deployment

```bash
# Authenticate
crewai login

# Create deployment (auto-detects repo, transfers .env vars securely)
crewai deploy create

# Monitor (first deploy takes 10-15 min)
crewai deploy status
crewai deploy logs

# Manage deployments
crewai deploy list              # List all deployments
crewai deploy push              # Push code updates
crewai deploy remove <id>       # Delete deployment
```

### Web Interface Deployment
1. Push code to GitHub
2. Log into https://app.crewai.com
3. Connect GitHub and select repository
4. Configure environment variables (KEY=VALUE, one per line)
5. Click Deploy and monitor via dashboard

### CI/CD API Deployment

Get a Personal Access Token from app.crewai.com → Settings → Account → Personal Access Token.
Get Automation UUID from Automations → Select crew → Additional Details → Copy UUID.

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_PERSONAL_ACCESS_TOKEN" \
     https://app.crewai.com/crewai_plus/api/v1/crews/YOUR-AUTOMATION-UUID/deploy
```

#### GitHub Actions Example
```yaml
name: Deploy CrewAI Automation
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger CrewAI Redeployment
        run: |
          curl -X POST \
               -H "Authorization: Bearer ${{ secrets.CREWAI_PAT }}" \
               https://app.crewai.com/crewai_plus/api/v1/crews/${{ secrets.CREWAI_AUTOMATION_UUID }}/deploy
```

### Project Structure Requirements for Deployment
- Entry point: `src/<project_name>/main.py`
- Crews must expose a `run()` function
- Flows must expose a `kickoff()` function
- All crew classes require `@CrewBase` decorator

### Deployed Automation REST API
| Endpoint | Purpose |
|----------|---------|
| `/inputs` | List required input parameters |
| `/kickoff` | Trigger execution with inputs |
| `/status/{kickoff_id}` | Check execution status |

### AMP Dashboard Tabs
- **Status**: Deployment info, API endpoint, auth token
- **Run**: Crew structure visualization
- **Executions**: Run history
- **Metrics**: Performance analytics
- **Traces**: Detailed execution insights

### Deployment Troubleshooting
| Error | Fix |
|-------|-----|
| Missing uv.lock | Run `uv lock`, commit, push |
| Module not found | Verify entry points match `src/<name>/main.py` structure |
| Crew not found | Ensure `@CrewBase` decorator on all crew classes |
| API key errors | Check env var names match code and are set in the platform |

---

## Environment Setup

### Required `.env`
```
OPENAI_API_KEY=sk-...
# Optional depending on tools/providers:
SERPER_API_KEY=...
ANTHROPIC_API_KEY=...
# Override default model:
MODEL=gpt-4o
```

### Python Version
Python >=3.10, <3.14

### Installation
```bash
uv tool install crewai        # Install CrewAI CLI
uv tool list                  # Verify installation
crewai create crew my_crew --skip-provider   # Scaffold a new project
crewai install                # Install project dependencies
crewai run                    # Execute
```

---

## Development Best Practices

1. **YAML-first configuration**: Define agents and tasks in YAML, keep crew classes minimal
2. **Check built-in tools** before writing custom ones
3. **Use structured output** (output_pydantic) for data that flows between tasks or crews
4. **Use guardrails** to validate task outputs programmatically
5. **Enable memory** for crews that benefit from cross-session learning
6. **Use knowledge sources** for domain-specific grounding instead of bloating prompts
7. **Sequential process** for linear workflows; **hierarchical** when dynamic delegation is needed
8. **Flows for multi-crew orchestration**: Use `@start`, `@listen`, `@router` for complex pipelines
9. **Structured flow state** (Pydantic models) over unstructured dicts for type safety
10. **Test with** `crewai test` to evaluate crew performance across iterations
11. **Verbose mode** during development, disable in production
12. **Rate limiting** (`max_rpm`) to avoid API throttling
13. **`respect_context_window=True`** to auto-handle token limits

## Common Pitfalls

- **Using `ChatOpenAI()`** — Always use `crewai.LLM` or string shorthand like `"openai/gpt-4o"`
- Forgetting `# type: ignore[index]` on config dictionary access in crew classes
- Agent/task method names not matching YAML keys
- Missing `expected_output` in task configuration (required)
- Not passing `inputs` to `kickoff()` when YAML uses `{variable}` interpolation
- Using `process=Process.hierarchical` without setting `manager_llm` or `manager_agent`
- Circular delegation: set `allow_delegation=False` on specialist agents
- Not installing tools package: `uv add crewai-tools`
