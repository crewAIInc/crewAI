<div align="center">

![Logo of CrewAI, two people rowing on a boat](./docs/crewai_logo.png)

# **CrewAI**

🤖 **CrewAI**: Cutting-edge framework for orchestrating role-playing and autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.

<h3>

[Homepage](https://www.crewai.io/) | [Documentation](https://docs.crewai.com/) | [Chat with the Docs](https://chatg.pt/DWjSBZn) | [Examples](https://github.com/joaomdmoura/crewai-examples) | [Discord](https://discord.com/invite/X4JWnZnxPb)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/joaomdmoura/crewAI)](https://github.com/joaomdmoura/crewAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## Table of contents

- [Why CrewAI?](#why-crewai)
- [Getting Started](#getting-started)
- [Key Features](#key-features)
- [Examples](#examples)
  - [Quick Tutorial](#quick-tutorial)
  - [Write Job Descriptions](#write-job-descriptions)
  - [Trip Planner](#trip-planner)
  - [Stock Analysis](#stock-analysis)
- [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
- [How CrewAI Compares](#how-crewai-compares)
- [Contribution](#contribution)
- [Telemetry](#telemetry)
- [License](#license)

## Why CrewAI?

The power of AI collaboration has too much to offer.
CrewAI is designed to enable AI agents to assume roles, share goals, and operate in a cohesive unit - much like a well-oiled crew. Whether you're building a smart assistant platform, an automated customer service ensemble, or a multi-agent research team, CrewAI provides the backbone for sophisticated multi-agent interactions.

## Getting Started

To get started with CrewAI, follow these simple steps:

### 1. Installation

```shell
pip install crewai
```

If you want to install the 'crewai' package along with its optional features that include additional tools for agents, you can do so by using the following command: pip install 'crewai[tools]'. This command installs the basic package and also adds extra components which require more dependencies to function."

```shell
pip install 'crewai[tools]'
```

### 2. Setting Up Your Crew

```python
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key

# You can choose to use a local model through Ollama for example. See https://docs.crewai.com/how-to/LLM-Connections/ for more information.

# os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
# os.environ["OPENAI_MODEL_NAME"] ='openhermes'  # Adjust based on available model
# os.environ["OPENAI_API_KEY"] ='sk-111111111111111111111111111111111111111111111111'

# You can pass an optional llm attribute specifying what model you wanna use.
# It can be a local model through Ollama / LM Studio or a remote
# model like OpenAI, Mistral, Anthropic or others (https://docs.crewai.com/how-to/LLM-Connections/)
#
# import os
# os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
#
# OR
#
# from langchain_openai import ChatOpenAI

search_tool = SerperDevTool()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  # You can pass an optional llm attribute specifying what model you wanna use.
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
  process = Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
```

In addition to the sequential process, you can use the hierarchical process to automatically assign a manager agent to a crew. The manager's responsibility is to properly coordinate the planning and execution of tasks through delegation and validation of results. [See more about the processes here](https://docs.crewai.com/core-concepts/Processes/).

## Key Features

- **Role-Based Agent Design**: Customize agents with specific roles, goals, and tools.
- **Autonomous Inter-Agent Delegation**: Agents can be configured to autonomously delegate tasks and inquire amongst themselves, enhancing problem-solving efficiency.
- **Flexible Task Management**: Define tasks with customizable tools and assign them to agents dynamically.
- **Processes Driven**: Currently supports `sequential` task execution and `hierarchical` processes. More complex processes like consensual and autonomous are being worked on.
- **Save output as file**: Save the output of individual tasks as a file, so you can use it later.
- **Parse output as Pydantic or JSON**: Parse the output of individual tasks as a Pydantic model or as JSON if you want to.
- **Works with Open Source Models**: Run your crew using OpenAI or open source models. Refer to the [Connect crewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring your agents' connections to models - even locally running models!

![CrewAI Mind Map](./docs/crewAI-mindmap.png "CrewAI Mind Map")

## Examples

You can test different real life examples of AI crews in the [crewAI-examples repo](https://github.com/joaomdmoura/crewAI-examples?tab=readme-ov-file):

- [Landing Page Generator](https://github.com/joaomdmoura/crewAI-examples/tree/main/landing_page_generator)
- [Having Human Input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
- [Trip Planner](https://github.com/joaomdmoura/crewAI-examples/tree/main/trip_planner)
- [Stock Analysis](https://github.com/joaomdmoura/crewAI-examples/tree/main/stock_analysis)

### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Write Job Descriptions

[Check out code for this example](https://github.com/joaomdmoura/crewAI-examples/tree/main/job-posting) or watch a video below:

[![Job postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Job postings")

### Trip Planner

[Check out code for this example](https://github.com/joaomdmoura/crewAI-examples/tree/main/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[Check out code for this example](https://github.com/joaomdmoura/crewAI-examples/tree/main/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

## Connecting Your Crew to a Model

CrewAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring you agents' connections to models.

## How CrewAI Compares

- **Autogen**: While Autogen does good in creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.

- **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

**CrewAI's Advantage**: CrewAI is built with production in mind. It offers the flexibility of Autogen's conversational agents and the structured process approach of ChatDev, but without the rigidity. CrewAI's processes are designed to be dynamic and adaptable, fitting seamlessly into both development and production workflows.

## Contribution

CrewAI is open-source and we encourage contributions. If you're interested in contributing, here's how you can do it:

1. Fork the repository.
2. Create a new branch for your feature or improvement.
3. Implement your changes.
4. Submit a pull request.

We value your input.

### Installing Dependencies

```bash
poetry lock
poetry install
```

### Virtual Env

```bash
poetry shell
```

### Pre-commit hooks

```bash
pre-commit install
```

### Running Tests

```bash
poetry run pytest
```

### Running static type checks

```bash
poetry run mypy
```

### Packaging

```bash
poetry build
```

### Installing Locally

```bash
pip install dist/*.tar.gz
```

## Telemetry

CrewAI collects anonymized usage data to aid in improving the library by prioritizing features, integrations, and tools that are most commonly used.

We do not collect data on the prompts, task descriptions, agent backstories, goals, tool usage, API calls, or responses. Additionally, we do not collect any data that is being processed by the agents, such as secrets or environment variables.

Data collected includes:

- The version of CrewAI being used
- The version of Python being used
- General operating system details (e.g. number of CPUs, macOS/Windows/Linux)
- The number of agents and tasks in a crew
- The process being used (sequential or hierarchical)
- The language model being used
- The roles of agents in a crew
- The names of available tools

This information helps us understand how users are utilizing CrewAI, which allows us to focus on the most common use cases, improve support for popular integrations, and develop tools and features that meet the needs of our users.

Users can opt to share the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews.

## License

CrewAI is released under the MIT License.
