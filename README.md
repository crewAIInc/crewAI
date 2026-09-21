<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>
<p align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <a href="https://trendshift.io/repositories/11239" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/11239" alt="crewAIInc%2FcrewAI | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <a href="https://crewai.com">Homepage</a>
  ·
  <a href="https://crewai.com/open-source">Open Source</a>
  ·
  <a href="https://docs.crewai.com">Docs</a>
  ·
  <a href="https://app.crewai.com">Start Cloud Trial</a>
  ·
  <a href="https://blog.crewai.com">Blog</a>
  ·
  <a href="https://community.crewai.com">Forum</a>
</p>

<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="https://img.shields.io/github/stars/crewAIInc/crewAI" alt="GitHub Repo stars">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/network/members">
    <img src="https://img.shields.io/github/forks/crewAIInc/crewAI" alt="GitHub forks">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/issues">
    <img src="https://img.shields.io/github/issues/crewAIInc/crewAI" alt="GitHub issues">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/pulls">
    <img src="https://img.shields.io/github/issues-pr/crewAIInc/crewAI" alt="GitHub pull requests">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/crewai/">
    <img src="https://img.shields.io/pypi/v/crewai" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/crewai/">
    <img src="https://img.shields.io/pypi/dm/crewai" alt="PyPI downloads">
  </a>
  <a href="https://twitter.com/crewAIInc">
    <img src="https://img.shields.io/twitter/follow/crewAIInc?style=social" alt="Twitter Follow">
  </a>
</p>

### Fast and Flexible Multi-Agent Automation Framework

> CrewAI is an open-source Python framework with high-level abstractions and low-level APIs for building production-ready multi-agent workflows.
> It gives developers autonomous agent collaboration through Crews and precise, event-driven control through Flows.

- **CrewAI Crews**: Optimize for autonomy and collaborative intelligence with role-based AI agents.
- **CrewAI Flows**: Build event-driven automations that combine precise workflow control, single LLM calls, and native support for Crews.

With over 100,000 developers certified through our community courses at [learn.crewai.com](https://learn.crewai.com), CrewAI is rapidly becoming the
standard for production-ready agentic automation.

# CrewAI AMP Suite

For organizations that need a commercial control plane around CrewAI, [CrewAI AMP Suite](https://crewai.com/amp) adds managed deployment, observability, governance, security, and enterprise support.

You can try one part of the suite, the [Crew Control Plane, for free](https://app.crewai.com).

## Crew Control Plane Key Features:

- **Tracing & Observability**: Monitor and track your AI agents and workflows in real-time, including metrics, logs, and traces.
- **Unified Control Plane**: A centralized platform for managing, monitoring, and scaling your AI agents and workflows.
- **Seamless Integrations**: Easily connect with existing enterprise systems, data sources, and cloud infrastructure.
- **Advanced Security**: Built-in robust security and compliance measures ensuring safe deployment and management.
- **Actionable Insights**: Real-time analytics and reporting to optimize performance and decision-making.
- **24/7 Support**: Dedicated enterprise support to ensure uninterrupted operation and quick resolution of issues.
- **On-premise and Cloud Deployment Options**: Deploy CrewAI AMP on-premise or in the cloud, depending on your security and compliance requirements.

CrewAI AMP is designed for enterprises seeking a powerful, reliable solution to transform complex business processes into efficient,
intelligent automations.

## Table of contents

- [Build with AI](#build-with-ai)
- [Why CrewAI?](#why-crewai)
- [Getting Started](#getting-started)
  - [Learning Resources](#learning-resources)
  - [Understanding Flows and Crews](#understanding-flows-and-crews)
  - [Installation](#1-installation)
  - [Setting Up Your Crew](#2-setting-up-your-crew)
  - [Running Your Crew](#3-running-your-crew)
- [Key Features](#key-features)
- [Examples](#examples)
  - [Quick Tutorial](#quick-tutorial)
  - [Write Job Descriptions](#write-job-descriptions)
  - [Trip Planner](#trip-planner)
  - [Stock Analysis](#stock-analysis)
  - [Using Crews and Flows Together](#using-crews-and-flows-together)
- [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
- [When to Use CrewAI](#when-to-use-crewai)
- [Contribution](#contribution)
- [Telemetry](#telemetry)
- [License](#license)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

## Build with AI

Using an AI coding agent? Teach it CrewAI best practices in one command:

**Claude Code:**
```shell
/plugin marketplace add crewAIInc/skills
/plugin install crewai-skills@crewai-plugins
/reload-plugins
```
Four skills that activate automatically when you ask relevant CrewAI questions:

| Skill | When it runs |
|-------|--------------|
| `getting-started` | Scaffolding new projects, choosing between `LLM.call()` / `Agent` / `Crew` / `Flow`, wiring `crew.jsonc` / `main.py` |
| `design-agent` | Configuring agents — role, goal, backstory, tools, LLMs, memory, guardrails |
| `design-task` | Writing task descriptions, dependencies, structured output (`output_pydantic`, `output_json`), human review |
| `ask-docs` | Querying the live [CrewAI docs MCP server](https://docs.crewai.com/mcp) for up-to-date API details |

**Cursor, Codex, Windsurf, and others ([skills.sh](https://skills.sh/crewaiinc/skills)):**
```shell
npx skills add crewaiinc/skills
```

This installs the official [CrewAI Skills](https://github.com/crewAIInc/skills) — structured instructions that teach coding agents how to scaffold Flows, configure Crews, design agents and tasks, and follow CrewAI patterns.

## Why CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI unlocks the true potential of multi-agent automation, delivering speed, flexibility, and control through Crews of AI agents and event-driven Flows:

- **Purpose-built architecture**: Designed specifically for agent orchestration, with a lightweight Python core and clean primitives for real-world automation.
- **High Performance**: Optimized for speed and minimal resource usage, enabling faster execution.
- **Flexible Low-Level Customization**: Complete freedom to customize everything from workflows and system architecture to agent behaviors, internal prompts, and execution logic.
- **Ideal for Every Use Case**: Proven effective for simple tasks, complex workflows, and production-grade automation.
- **Robust Community**: Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.

CrewAI empowers developers and teams to build intelligent automations that balance simplicity, flexibility, and production-grade control.

## Getting Started

Setup and run your first CrewAI agents by following this tutorial.

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

Learn CrewAI through our comprehensive courses:

- [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Master the fundamentals of multi-agent systems
- [Practical Multi AI Agents and Advanced Use Cases](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) - Deep dive into advanced implementations

### Understanding Flows and Crews

CrewAI offers two powerful, complementary approaches that work seamlessly together to build sophisticated AI applications:

1. **Crews**: Teams of AI agents with true autonomy and agency, working together to accomplish complex tasks through role-based collaboration. Crews enable:

   - Natural, autonomous decision-making between agents
   - Dynamic task delegation and collaboration
   - Specialized roles with defined goals and expertise
   - Flexible problem-solving approaches

2. **Flows**: Production-ready, event-driven workflows that deliver precise control over complex automations. Flows provide:

   - Fine-grained control over execution paths for real-world scenarios
   - Secure, consistent state management between tasks
   - Clean integration of AI agents with production Python code
   - Conditional branching for complex business logic

The true power of CrewAI emerges when combining Crews and Flows. This synergy allows you to:

- Build complex, production-grade applications
- Balance autonomy with precise control
- Handle sophisticated real-world scenarios
- Maintain clean, maintainable code structure

### Getting Started with Installation

To get started with CrewAI, follow these simple steps. The full walkthrough lives in the [installation guide](https://docs.crewai.com/en/installation).

### 1. Installation

CrewAI requires `Python >=3.10 and <3.14`. Check your version with:

```bash
python3 --version
```

CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling. If you haven't installed `uv` yet, install it first.

**macOS/Linux:**

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your system doesn't have `curl`, you can use `wget`:

```shell
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows:**

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you run into any issues, refer to [UV's installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Then install the CrewAI CLI:

```shell
uv tool install crewai
```

If you encounter a `PATH` warning, run:

```shell
uv tool update-shell
```

If you encounter the `chroma-hnswlib==0.7.6` build error (`fatal error C1083: Cannot open include file: 'float.h'`) on Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with *Desktop development with C++*.

Verify the install:

```shell
uv tool list
```

You should see something like:

```shell
crewai v0.102.0
- crewai
```

To upgrade the global CLI later:

```shell
uv tool install crewai --upgrade
```

This upgrades the **global `crewai` CLI tool** only. To upgrade the `crewai` version inside a project's virtual environment, see [Upgrading CrewAI in a project](https://docs.crewai.com/en/guides/migration/upgrading-crewai).

### 2. Setting Up Your Crew

`crewai create crew` creates a JSON-first crew project. Agents live in `agents/*.jsonc`, tasks and crew-level settings live in `crew.jsonc`, and `crewai run` loads that JSON definition directly.

```shell
crewai create crew <project_name>
```

This command creates a new project folder with the following structure:

```
my_project/
├── .gitignore
├── .env
├── agents/
│   └── researcher.jsonc
├── crew.jsonc
├── knowledge/
├── pyproject.toml
├── README.md
├── skills/
└── tools/
```

If you need the older Python/YAML scaffold with `crew.py`, `config/agents.yaml`, and `config/tasks.yaml`, run:

```shell
crewai create crew <project_name> --classic
```

See [Using Annotations](https://docs.crewai.com/en/learn/using-annotations) for the classic pattern.

#### To customize your project, you can:

- Modify `agents/*.jsonc` to define each agent's role, goal, backstory, LLM, tools, and behavior.
- Modify `crew.jsonc` to define tasks, process, and input defaults.
- Add custom tools in `tools/` and reference them as `"custom:<name>"`.
- Add optional knowledge files in `knowledge/` and skill files in `skills/`.
- Add your environment variables into the `.env` file.

Use `{placeholder}` values in agent and task text, then set defaults in `crew.jsonc` under `inputs`. When you run `crewai run`, the CLI prompts for any missing values.

#### Example of a simple crew with a sequential process:

```shell
crewai create crew latest-ai-development
cd latest_ai_development
```

Then edit the generated files:

**agents/researcher.jsonc**

```jsonc
{
  "role": "{topic} Senior Data Researcher",
  "goal": "Uncover cutting-edge developments in {topic}",
  "backstory": "You're a seasoned researcher who finds relevant information and presents it clearly.",
  "llm": "openai/gpt-4o",
  "tools": ["SerperDevTool"],
  "settings": {
    "verbose": true
  }
}
```

**agents/reporting_analyst.jsonc**

```jsonc
{
  "role": "{topic} Reporting Analyst",
  "goal": "Create detailed reports based on {topic} data analysis and research findings",
  "backstory": "You're a meticulous analyst who turns complex data into clear, concise reports.",
  "llm": "openai/gpt-4o",
  "settings": {
    "verbose": true
  }
}
```

**crew.jsonc**

```jsonc
{
  "name": "Latest AI Development",
  "agents": ["researcher", "reporting_analyst"],
  "tasks": [
    {
      "name": "research_task",
      "description": "Conduct thorough research about {topic}. Find recent, relevant information.",
      "expected_output": "A list with 10 bullet points of the most relevant information about {topic}.",
      "agent": "researcher"
    },
    {
      "name": "reporting_task",
      "description": "Review the research and expand each topic into a full section for a report.",
      "expected_output": "A markdown report with the main topics, each with a full section of information. No fenced code blocks around the whole document.",
      "agent": "reporting_analyst",
      "context": ["research_task"],
      "output_file": "output/report.md",
      "markdown": true
    }
  ],
  "process": "sequential",
  "verbose": true,
  "inputs": {
    "topic": "AI Agents"
  }
}
```

### 3. Running Your Crew

Before running your crew, set the required keys in your `.env` file:

- Your model provider API key — see [LLM setup](https://docs.crewai.com/en/concepts/llms#setting-up-your-llm)
- A [Serper.dev](https://serper.dev/) API key if you use web search: `SERPER_API_KEY=YOUR_KEY_HERE`

Then install dependencies and run from the project directory:

```shell
crewai install
crewai run
```

If you need additional packages, use `uv add <package-name>`.

You should see the output in the console, and `output/report.md` should be created in the project root.

In addition to the sequential process, you can use the hierarchical process, which automatically assigns a manager to the defined crew to properly coordinate the planning and execution of tasks through delegation and validation of results. [See more about the processes here](https://docs.crewai.com/en/concepts/processes).

For a Flow-first walkthrough, see the [Quickstart](https://docs.crewai.com/en/quickstart).

## Key Features

CrewAI gives developers a practical foundation for building agentic systems that move from prototype to production: autonomous collaboration where it helps, explicit workflow control where it matters, and Python-native customization throughout.

- **Crews for autonomy**: Model teams of specialized AI agents with roles, goals, tools, and tasks.
- **Flows for control**: Build event-driven workflows with state, branching, routing, and production logic.
- **Seamless integration**: Combine Crews and Flows to create complex, real-world automations.
- **Python-native customization**: Customize prompts, tools, execution paths, state, and integrations without fighting the framework.
- **Agent-ready capabilities**: Use tools, memory, knowledge, checkpointing, async execution, and MCP/A2A support for more capable production agents.
- **Production-ready patterns**: Add deterministic steps, human input, structured outputs, and checkpointing as your system grows.
- **Thriving community**: Backed by robust documentation and over 100,000 certified developers, providing exceptional support and guidance.

Choose CrewAI to build powerful, adaptable, and production-ready AI automations.

## Examples

You can test different real life examples of AI crews in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

- [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator)
- [Having Human input on the execution](https://docs.crewai.com/en/learn/human-input-on-execution)
- [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner)
- [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis)

### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Write Job Descriptions

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### Trip Planner

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

### Using Crews and Flows Together

CrewAI's power truly shines when combining Crews with Flows to create sophisticated automation pipelines.
CrewAI flows support logical operators like `or_` and `and_` to combine multiple conditions. This can be used with `@start`, `@listen`, or `@router` decorators to create complex triggering conditions.

- `or_`: Triggers when any of the specified conditions are met.
- `and_`: Triggers when all of the specified conditions are met.

Here's how you can orchestrate multiple Crews within a Flow:

```python
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import Crew, Agent, Task, Process
from pydantic import BaseModel

# Define structured state for precise control
class MarketState(BaseModel):
    sentiment: str = "neutral"
    confidence: float = 0.0
    recommendations: list = []

class AdvancedAnalysisFlow(Flow[MarketState]):
    @start()
    def fetch_market_data(self):
        # Demonstrate low-level control with structured state
        self.state.sentiment = "analyzing"
        return {"sector": "tech", "timeframe": "1W"}  # These parameters match the task description template

    @listen(fetch_market_data)
    def analyze_with_crew(self, market_data):
        # Show crew agency through specialized roles
        analyst = Agent(
            role="Senior Market Analyst",
            goal="Conduct deep market analysis with expert insight",
            backstory="You're a veteran analyst known for identifying subtle market patterns"
        )
        researcher = Agent(
            role="Data Researcher",
            goal="Gather and validate supporting market data",
            backstory="You excel at finding and correlating multiple data sources"
        )

        analysis_task = Task(
            description="Analyze {sector} sector data for the past {timeframe}",
            expected_output="Detailed market analysis with confidence score",
            agent=analyst
        )
        research_task = Task(
            description="Find supporting data to validate the analysis",
            expected_output="Corroborating evidence and potential contradictions",
            agent=researcher
        )

        # Demonstrate crew autonomy
        analysis_crew = Crew(
            agents=[analyst, researcher],
            tasks=[analysis_task, research_task],
            process=Process.sequential,
            verbose=True
        )
        return analysis_crew.kickoff(inputs=market_data)  # Pass market_data as named inputs

    @router(analyze_with_crew)
    def determine_next_steps(self):
        # Show flow control with conditional routing
        if self.state.confidence > 0.8:
            return "high_confidence"
        elif self.state.confidence > 0.5:
            return "medium_confidence"
        return "low_confidence"

    @listen("high_confidence")
    def execute_strategy(self):
        # Demonstrate complex decision making
        strategy_crew = Crew(
            agents=[
                Agent(role="Strategy Expert",
                      goal="Develop optimal market strategy")
            ],
            tasks=[
                Task(description="Create detailed strategy based on analysis",
                     expected_output="Step-by-step action plan")
            ]
        )
        return strategy_crew.kickoff()

    @listen(or_("medium_confidence", "low_confidence"))
    def request_additional_analysis(self):
        self.state.recommendations.append("Gather more data")
        return "Additional analysis required"
```

This example demonstrates how to:

1. Use Python code for basic data operations
2. Create and execute Crews as steps in your workflow
3. Use Flow decorators to manage the sequence of operations
4. Implement conditional branching based on Crew results

## Connecting Your Crew to a Model

CrewAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/en/learn/llm-connections) page for details on configuring your agents' connections to models.

## When to Use CrewAI

Use CrewAI when you need more than a single prompt or chatbot: multi-step work, specialized agents, tool use, structured outputs, human review, or workflows that combine autonomous reasoning with explicit business logic.

CrewAI is especially useful when you want to:

- Coordinate multiple agents with clear roles and tasks.
- Wrap agent work in deterministic, event-driven workflows.
- Keep application logic in regular Python.
- Move from experiment to production without changing frameworks.
- Add tools, memory, checkpointing, and async execution as your system grows.

## Contribution

CrewAI is open-source and we welcome contributions. See
[`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md) for the full setup guide,
branching conventions, and PR checklist.

Quick start:

```bash
git clone https://github.com/crewAIInc/crewAI.git
cd crewAI
uv sync --all-groups --all-extras
uv run pre-commit install
```

```bash
# Tests
uv run pytest lib/crewai/tests/ -x -q

# Type checks
uv run mypy lib/
```

### Contributing to the docs

The site at [docs.crewai.com](https://docs.crewai.com) is published from
`docs/` by [Mintlify](https://www.mintlify.com/). The docs use directory-based
versioning: edits to `docs/edge/<lang>/...` (e.g.
`docs/edge/en/concepts/agents.mdx`) land under the **Edge** version selector
immediately and are frozen into a new versioned snapshot under
`docs/v<X.Y.Z>/` at the next release cut. Frozen snapshots are immutable — CI
rejects PRs that modify them without a `[docs-freeze]` title prefix. The
release CLI (`devtools release`) handles the freeze automatically; see
[`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md) for contributor guidance and
[`lib/devtools/README.md`](lib/devtools/README.md) for release tooling.

## Telemetry

CrewAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.

It's pivotal to understand that **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned. When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:

- Version of CrewAI
  - So we can understand how many users are using the latest version
- Version of Python
  - So we can decide on what versions to better support
- General OS (e.g. number of CPUs, macOS/Windows/Linux)
  - So we know what OS we should focus on and if we could build specific OS related features
- Number of agents and tasks in a crew
  - So we make sure we are testing internally with similar use cases and educate people on the best practices
- Crew Process being used
  - Understand where we should focus our efforts
- If Agents are using memory or allowing delegation
  - Understand if we improved the features or maybe even drop them
- If Tasks are being executed in parallel or sequentially
  - Understand if we should focus more on parallel execution
- Language model being used
  - Improved support on most used languages
- Roles of agents in a crew
  - Understand high level use cases so we can build better tools, integrations and examples about it
- Tools names available
  - Understand out of the publicly available tools, which ones are being used the most so we can improve them

Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

### General

- [What exactly is CrewAI?](#q-what-exactly-is-crewai)
- [How do I install CrewAI?](#q-how-do-i-install-crewai)
- [Is CrewAI a standalone framework?](#q-is-crewai-a-standalone-framework)
- [Is CrewAI open-source?](#q-is-crewai-open-source)
- [Does CrewAI collect data from users?](#q-does-crewai-collect-data-from-users)

### Features and Capabilities

- [Can CrewAI handle complex use cases?](#q-can-crewai-handle-complex-use-cases)
- [Can I use CrewAI with local AI models?](#q-can-i-use-crewai-with-local-ai-models)
- [What makes Crews different from Flows?](#q-what-makes-crews-different-from-flows)
- [Does CrewAI support fine-tuning or training custom models?](#q-does-crewai-support-fine-tuning-or-training-custom-models)

### Resources and Community

- [Where can I find real-world CrewAI examples?](#q-where-can-i-find-real-world-crewai-examples)
- [How can I contribute to CrewAI?](#q-how-can-i-contribute-to-crewai)

### Enterprise Features

- [What additional features does CrewAI AMP offer?](#q-what-additional-features-does-crewai-amp-offer)
- [Is CrewAI AMP available for cloud and on-premise deployments?](#q-is-crewai-amp-available-for-cloud-and-on-premise-deployments)
- [Can I try CrewAI AMP for free?](#q-can-i-try-crewai-amp-for-free)

### Q: What exactly is CrewAI?

A: CrewAI is a lean, fast Python framework built specifically for orchestrating autonomous AI agents and production-ready agentic workflows.

### Q: How do I install CrewAI?

A: Install the CrewAI CLI with [UV](https://docs.astral.sh/uv/):

```shell
uv tool install crewai
```

Then create a project with `crewai create crew <project_name>`, run `crewai install`, and start it with `crewai run`. See the [installation guide](https://docs.crewai.com/en/installation) for details.

### Q: Is CrewAI a standalone framework?

A: Yes. CrewAI is a standalone Python framework with its own primitives for agents, tasks, crews, flows, tools, and orchestration.

### Q: Can CrewAI handle complex use cases?

A: Yes. CrewAI excels at both simple and highly complex real-world scenarios, offering deep customization options at both high and low levels, from internal prompts to sophisticated workflow orchestration.

### Q: Can I use CrewAI with local AI models?

A: Absolutely! CrewAI supports various language models, including local ones. Tools like Ollama and LM Studio allow seamless integration. Check the [LLM Connections documentation](https://docs.crewai.com/en/learn/llm-connections) for more details.

### Q: What makes Crews different from Flows?

A: Crews provide autonomous agent collaboration, ideal for tasks requiring flexible decision-making and dynamic interaction. Flows offer precise, event-driven control, ideal for managing detailed execution paths and secure state management. You can seamlessly combine both for maximum effectiveness.

### Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source and actively encourages community contributions and collaboration.

### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry data strictly for improvement purposes. Sensitive data such as prompts, tasks, or API responses are never collected unless explicitly enabled by the user.

### Q: Where can I find real-world CrewAI examples?

A: Check out practical examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples), covering use cases like trip planners, stock analysis, and job postings.

### Q: How can I contribute to CrewAI?

A: Contributions are warmly welcomed! Fork the repository, create your branch, implement your changes, and submit a pull request. See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md) for detailed guidelines.

### Q: What additional features does CrewAI AMP offer?

A: CrewAI AMP provides advanced features such as a unified control plane, real-time observability, secure integrations, advanced security, actionable insights, and dedicated 24/7 enterprise support.

### Q: Is CrewAI AMP available for cloud and on-premise deployments?

A: Yes, CrewAI AMP supports both cloud-based and on-premise deployment options, allowing enterprises to meet their specific security and compliance requirements.

### Q: Can I try CrewAI AMP for free?

A: Yes, you can explore part of the CrewAI AMP Suite by accessing the [Crew Control Plane](https://app.crewai.com) for free.

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, CrewAI can integrate with custom-trained or fine-tuned models, allowing you to enhance your agents with domain-specific knowledge and accuracy.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Absolutely! CrewAI agents can easily integrate with external tools, APIs, and databases, empowering them to leverage real-world data and resources.

### Q: Is CrewAI suitable for production environments?

A: Yes, CrewAI is designed with production-grade patterns that support reliable, stable, and scalable agentic workflows.

### Q: How scalable is CrewAI?

A: CrewAI is highly scalable, supporting simple automations and large-scale workflows involving numerous agents and complex tasks simultaneously.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI AMP includes advanced debugging, tracing, and real-time observability features, simplifying the management and troubleshooting of your automations.

### Q: What programming languages does CrewAI support?

A: CrewAI is primarily Python-based but easily integrates with services and APIs written in any programming language through its flexible API integration capabilities.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, CrewAI provides extensive beginner-friendly tutorials, courses, and documentation through learn.crewai.com, supporting developers at all skill levels.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes, CrewAI fully supports human-in-the-loop workflows, allowing seamless collaboration between human experts and AI agents for enhanced decision-making.
