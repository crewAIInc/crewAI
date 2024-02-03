<div align="center">

![Logo of crewAI, two people rowing on a boat](./docs/crewai_logo.png)

# **crewAI**

🤖 **crewAI**: Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.

<h3>

[Homepage](https://www.crewai.io/) | [Documentation](https://joaomdmoura.github.io/crewAI/) | [Chat with Docs](https://chatg.pt/DWjSBZn) | [Examples](https://joaomdmoura.github.io/crewAI/#examples-and-tutorials) | [Discord](https://discord.com/invite/X4JWnZnxPb)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/joaomdmoura/crewAI)](https://github.com/joaomdmoura/crewAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<!-- [![Discord](https://img.shields.io/discord/1192246288507474000)](https://discord.com/invite/X4JWnZnxPb) -->

</div>

## Table of contents

- [Why CrewAI?](#why-crewai)
- [Getting Started](#getting-started)
- [Key Features](#key-features)
- [Examples](#examples)
  - [Quick Tutorial](#quick-tutorial)
  - [Trip Planner](#trip-planner)
  - [Stock Analysis](#stock-analysis)
- [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
- [How CrewAI Compares](#how-crewai-compares)
- [Contribution](#contribution)
- [Hire CrewAI](#hire-crewai)
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

The example below also uses DuckDuckGo's Search. You can install it with `pip` too:

```shell
pip install duckduckgo-search
```

### 2. Setting Up Your Crew

```python
import os
from crewai import Agent, Task, Crew, Process

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# You can choose to use a local model through Ollama for example. See ./docs/how-to/llm-connections.md for more information.
# from langchain.llms import Ollama
# ollama_llm = Ollama(model="openhermes")

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
  # You can pass an optional llm attribute specifying what mode you wanna use.
  # It can be a local model through Ollama / LM Studio or a remote
  # model like OpenAI, Mistral, Antrophic or others (https://python.langchain.com/docs/integrations/llms/)
  #
  # Examples:
  # llm=ollama_llm # was defined above in the file
  # llm=OpenAI(model_name="gpt-3.5", temperature=0.7)
  # For the OpenAI model you would need to import
  # from langchain_openai import OpenAI
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  # (optional) llm=ollama_llm
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.""",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
```

In addition to the sequential process, you can use the hierarchical process, which automatically assigns a manager to the defined crew to properly coordinate the planning and execution of tasks through delegation and validation of results. See more about the processes [here](./docs/core-concepts/Managing-Processes.md).

## Key Features

- **Role-Based Agent Design**: Customize agents with specific roles, goals, and tools.
- **Autonomous Inter-Agent Delegation**: Agents can autonomously delegate tasks and inquire amongst themselves, enhancing problem-solving efficiency.
- **Flexible Task Management**: Define tasks with customizable tools and assign them to agents dynamically.
- **Processes Driven**: Currently only supports `sequential` task execution and `hierarchical` processes, but more complex processes like consensual and autonomous are being worked on.
- **Works with Open Source Models**: Run your crew using Open AI or open source models refer to the [Connect crewAI to LLMs](https://joaomdmoura.github.io/crewAI/how-to/LLM-Connections/) page for details on configuring you agents' connections to models, even ones running locally!

![CrewAI Mind Map](./docs/crewAI-mindmap.png "CrewAI Mind Map")

## Examples

You can test different real life examples of AI crews in the [ `crewAI-examples`](https://github.com/joaomdmoura/crewAI-examples?tab=readme-ov-file) repo:

- [Landing Page Generator](https://github.com/joaomdmoura/crewAI-examples/tree/main/landing_page_generator)
- [Having Human input on the execution](./docs/how-to/Human-Input-on-Execution.md)
- [Trip Planner](https://github.com/joaomdmoura/crewAI-examples/tree/main/trip_planner)
- [Stock Analysis](https://github.com/joaomdmoura/crewAI-examples/tree/main/stock_analysis)

### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Trip Planner

[Check out code for this example](https://github.com/joaomdmoura/crewAI-examples/tree/main/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[Check out code for this example](https://github.com/joaomdmoura/crewAI-examples/tree/main/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

## Connecting Your Crew to a Model

crewAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect crewAI to LLMs](https://joaomdmoura.github.io/crewAI/how-to/LLM-Connections/) page for details on configuring you agents' connections to models.

## How CrewAI Compares

- **Autogen**: While Autogen excels in creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.

- **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

**CrewAI's Advantage**: CrewAI is built with production in mind. It offers the flexibility of Autogen's conversational agents and the structured process approach of ChatDev, but without the rigidity. CrewAI's processes are designed to be dynamic and adaptable, fitting seamlessly into both development and production workflows.

## Contribution

CrewAI is open-source and we welcome contributions. If you're looking to contribute, please:

- Fork the repository.
- Create a new branch for your feature.
- Add your feature or improvement.
- Send a pull request.
- We appreciate your input!

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
poetry run pyright
```

### Packaging

```bash
poetry build
```

### Installing Locally

```bash
pip install dist/*.tar.gz
```

## Hire CrewAI

We're a company developing crewAI and crewAI Enterprise, we for a limited time are offer consulting with selected customers, to get them early access to our enterprise solution
If you are interested on having access to it and hiring weekly hours with our team, feel free to email us at [sales@crewai.io](mailto:sales@crewai.io).

## License

CrewAI is released under the MIT License.
