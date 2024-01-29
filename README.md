THIS IS A TEST

# crewAI

![Logo of crewAI, tow people rowing on a boat](./docs/crewai_logo.png)

🤖 Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.

- [crewAI](#crewai)
  - [Why CrewAI?](#why-crewai)
  - [Getting Started](#getting-started)
  - [Key Features](#key-features)
  - [Examples](#examples)
    - [Code](#code)
    - [Video](#video)
      - [Quick Tutorial](#quick-tutorial)
      - [Trip Planner](#trip-planner)
      - [Stock Analysis](#stock-analysis)
  - [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
  - [How CrewAI Compares](#how-crewai-compares)
  - [Contribution](#contribution)
    - [Installing Dependencies](#installing-dependencies)
    - [Virtual Env](#virtual-env)
    - [Pre-commit hooks](#pre-commit-hooks)
    - [Running Tests](#running-tests)
    - [Packaging](#packaging)
    - [Installing Locally](#installing-locally)
  - [Hire CrewAI](#hire-crewai)
  - [License](#license)

## Why CrewAI?

The power of AI collaboration has too much to offer.
CrewAI is designed to enable AI agents to assume roles, share goals, and operate in a cohesive unit - much like a well-oiled crew. Whether you're building a smart assistant platform, an automated customer service ensemble, or a multi-agent research team, CrewAI provides the backbone for sophisticated multi-agent interactions.

- 🤖 [Talk with the Docs](https://chatg.pt/DWjSBZn)
- 📄 [Documentation Wiki](https://joaomdmoura.github.io/crewAI/)

## Getting Started

To get started with CrewAI, follow these simple steps:

1. **Installation**:

```shell
pip install crewai
```

The example below also uses duckduckgo, so also install that
```shell
pip install duckduckgo-search
```

2. **Setting Up Your Crew**:

```python
import os
from crewai import Agent, Task, Crew, Process

os.environ["OPENAI_API_KEY"] = "YOUR KEY"

# You can choose to use a local model through Ollama for example. See ./docs/llm-connections.md for more information.
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
  You have a knack for dissecting complex data and presenting
  actionable insights.""",
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
  backstory="""You are a renowned Content Strategist, known for
  your insightful and engaging articles.
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

Currently the only supported process is `Process.sequential`, where one task is executed after the other and the outcome of one is passed as extra content into this next.


## Key Features

- **Role-Based Agent Design**: Customize agents with specific roles, goals, and tools.
- **Autonomous Inter-Agent Delegation**: Agents can autonomously delegate tasks and inquire amongst themselves, enhancing problem-solving efficiency.
- **Flexible Task Management**: Define tasks with customizable tools and assign them to agents dynamically.
- **Processes Driven**: Currently only supports `sequential` task execution but more complex processes like consensual and hierarchical being worked on.
- **Works with Open Source Models**: Run your crew using Open AI or open source models refer to the [Connect crewAI to LLMs](./docs/llm-connections.md) page for details on configuring you agents' connections to models, even ones running locally!

![CrewAI Mind Map](./docs/crewAI-mindmap.png "CrewAI Mind Map")

## Examples
You can test different real life examples of AI crews [in the examples repo](https://github.com/joaomdmoura/crewAI-examples?tab=readme-ov-file)

### Code
- [Trip Planner](https://github.com/joaomdmoura/crewAI-examples/tree/main/trip_planner)
- [Stock Analysis](https://github.com/joaomdmoura/crewAI-examples/tree/main/stock_analysis)
- [Landing Page Generator](https://github.com/joaomdmoura/crewAI-examples/tree/main/landing_page_generator)
- [Having Human input on the execution](./docs/how-to/Human-Input-on-Execution.md)

### Video
#### Quick Tutorial
[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/0.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

#### Trip Planner
[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/0.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

#### Stock Analysis
[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/0.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

## Connecting Your Crew to a Model

crewAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect crewAI to LLMs](./docs/how-to/llm-connections.md) page for details on configuring you agents' connections to models.

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
If you are interested on having access to it and hiring weekly hours with our team, feel free to email us at [sales@crewai.io](mailto:sales@crewai.io)

## License
CrewAI is released under the MIT License
