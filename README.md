<div align="center">

![Logo of CrewAI, two people rowing on a boat](./docs/crewai_logo.png)

# **CrewAI**

🤖 **CrewAI**: Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.

<h3>

[Homepage](https://www.crewai.com/) | [Documentation](https://docs.crewai.com/) | [Chat with Docs](https://chatg.pt/DWjSBZn) | [Examples](https://github.com/crewAIInc/crewAI-examples) | [Discourse](https://community.crewai.com)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/joaomdmoura/crewAI)](https://github.com/crewAIInc/crewAI)
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

Ensure you have Python >=3.10 <=3.13 installed on your system. CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, install CrewAI:

```shell
pip install crewai
```

If you want to install the 'crewai' package along with its optional features that include additional tools for agents, you can do so by using the following command:

```shell
pip install 'crewai[tools]'
```
The command above installs the basic package and also adds extra components which require more dependencies to function.

### 2. Setting Up Your Crew with the YAML Configuration

To create a new CrewAI project, run the following CLI (Command Line Interface) command:

```shell
crewai create crew <project_name>
```

This command creates a new project folder with the following structure:

```
my_project/
├── .gitignore
├── pyproject.toml
├── README.md
├── .env
└── src/
    └── my_project/
        ├── __init__.py
        ├── main.py
        ├── crew.py
        ├── tools/
        │   ├── custom_tool.py
        │   └── __init__.py
        └── config/
            ├── agents.yaml
            └── tasks.yaml
```

You can now start developing your crew by editing the files in the `src/my_project` folder. The `main.py` file is the entry point of the project, the `crew.py` file is where you define your crew, the `agents.yaml` file is where you define your agents, and the `tasks.yaml` file is where you define your tasks.

#### To customize your project, you can:

- Modify `src/my_project/config/agents.yaml` to define your agents.
- Modify `src/my_project/config/tasks.yaml` to define your tasks.
- Modify `src/my_project/crew.py` to add your own logic, tools, and specific arguments.
- Modify `src/my_project/main.py` to add custom inputs for your agents and tasks.
- Add your environment variables into the `.env` file.

#### Example of a simple crew with a sequential process:

Instatiate your crew:

```shell
crewai create crew latest-ai-development
```

Modify the files as needed to fit your use case:

**agents.yaml**

```yaml
# src/my_project/config/agents.yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.
      
reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} data analysis and research findings
  backstory: >
    You're a meticulous analyst with a keen eye for detail. You're known for
    your ability to turn complex data into clear and concise reports, making
    it easy for others to understand and act on the information you provide.
```

**tasks.yaml**

```yaml
# src/my_project/config/tasks.yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information given
    the current year is 2024.
  expected_output: >
    A list with 10 bullet points of the most relevant information about {topic}
  agent: researcher

reporting_task:
  description: >
    Review the context you got and expand each topic into a full section for a report.
    Make sure the report is detailed and contains any and all relevant information.
  expected_output: >
    A fully fledge reports with the mains topics, each with a full section of information.
    Formatted as markdown without '```'
  agent: reporting_analyst
  output_file: report.md
```

**crew.py**

```python
# src/my_project/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

@CrewBase
class LatestAiDevelopmentCrew():
	"""LatestAiDevelopment crew"""

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			tools=[SerperDevTool()]
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the LatestAiDevelopment crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		) 
```

**main.py**

```python
#!/usr/bin/env python
# src/my_project/main.py
import sys
from latest_ai_development.crew import LatestAiDevelopmentCrew

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI Agents'
    }
    LatestAiDevelopmentCrew().crew().kickoff(inputs=inputs)
```

### 3. Running Your Crew

Before running your crew, make sure you have the following keys set as environment variables in your `.env` file:

- An [OpenAI API key](https://platform.openai.com/account/api-keys) (or other LLM API key): `OPENAI_API_KEY=sk-...`
- A [Serper.dev](https://serper.dev/) API key: `SERPER_API_KEY=YOUR_KEY_HERE`

Lock the dependencies and install them by using the CLI command but first, navigate to your project directory:

```shell
cd my_project
crewai install (Optional)
```

To run your crew, execute the following command in the root of your project:

```bash
crewai run
```

or

```bash
python src/my_project/main.py
```

If an error happens due to the usage of poetry, please run the following command to update your crewai package:

```bash
crewai update
```

You should see the output in the console and the `report.md` file should be created in the root of your project with the full final report.

In addition to the sequential process, you can use the hierarchical process, which automatically assigns a manager to the defined crew to properly coordinate the planning and execution of tasks through delegation and validation of results. [See more about the processes here](https://docs.crewai.com/core-concepts/Processes/).

## Key Features

- **Role-Based Agent Design**: Customize agents with specific roles, goals, and tools.
- **Autonomous Inter-Agent Delegation**: Agents can autonomously delegate tasks and inquire amongst themselves, enhancing problem-solving efficiency.
- **Flexible Task Management**: Define tasks with customizable tools and assign them to agents dynamically.
- **Processes Driven**: Currently only supports `sequential` task execution and `hierarchical` processes, but more complex processes like consensual and autonomous are being worked on.
- **Save output as file**: Save the output of individual tasks as a file, so you can use it later.
- **Parse output as Pydantic or Json**: Parse the output of individual tasks as a Pydantic model or as a Json if you want to.
- **Works with Open Source Models**: Run your crew using Open AI or open source models refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring your agents' connections to models, even ones running locally!

![CrewAI Mind Map](./docs/crewAI-mindmap.png "CrewAI Mind Map")

## Examples

You can test different real life examples of AI crews in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

- [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
- [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
- [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
- [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Write Job Descriptions

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### Trip Planner

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

## Connecting Your Crew to a Model

CrewAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring you agents' connections to models.

## How CrewAI Compares

**CrewAI's Advantage**: CrewAI is built with production in mind. It offers the flexibility of Autogen's conversational agents and the structured process approach of ChatDev, but without the rigidity. CrewAI's processes are designed to be dynamic and adaptable, fitting seamlessly into both development and production workflows.

- **Autogen**: While Autogen does good in creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.

- **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

## Contribution

CrewAI is open-source and we welcome contributions. If you're looking to contribute, please:

- Fork the repository.
- Create a new branch for your feature.
- Add your feature or improvement.
- Send a pull request.
- We appreciate your input!

### Installing Dependencies

```bash
uv lock
uv sync
```

### Virtual Env

```bash
uv venv
```

### Pre-commit hooks

```bash
pre-commit install
```

### Running Tests

```bash
uv run pytest .
```

### Running static type checks

```bash
uvx mypy
```

### Packaging

```bash
uv build
```

### Installing Locally

```bash
pip install dist/*.tar.gz
```

## Telemetry

CrewAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.

It's pivotal to understand that **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned. When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. We don't offer a way to disable it now, but we will in the future.

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
  - Understand out of the publically available tools, which ones are being used the most so we can improve them

Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

### Q: What is CrewAI?
A: CrewAI is a cutting-edge framework for orchestrating role-playing, autonomous AI agents. It enables agents to work together seamlessly, tackling complex tasks through collaborative intelligence.

### Q: How do I install CrewAI?
A: You can install CrewAI using pip:
```shell
pip install crewai
```
For additional tools, use:
```shell
pip install 'crewai[tools]'
```

### Q: Can I use CrewAI with local models?
A: Yes, CrewAI supports various LLMs, including local models. You can configure your agents to use local models via tools like Ollama & LM Studio. Check the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/) for more details.

### Q: What are the key features of CrewAI?
A: Key features include role-based agent design, autonomous inter-agent delegation, flexible task management, process-driven execution, output saving as files, and compatibility with both open-source and proprietary models.

### Q: How does CrewAI compare to other AI orchestration tools?
A: CrewAI is designed with production in mind, offering flexibility similar to Autogen's conversational agents and structured processes like ChatDev, but with more adaptability for real-world applications.

### Q: Is CrewAI open-source?
A: Yes, CrewAI is open-source and welcomes contributions from the community.

### Q: Does CrewAI collect any data?
A: CrewAI uses anonymous telemetry to collect usage data for improvement purposes. No sensitive data (like prompts, task descriptions, or API calls) is collected. Users can opt-in to share more detailed data by setting `share_crew=True` on their Crews.

### Q: Where can I find examples of CrewAI in action?
A: You can find various real-life examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples), including trip planners, stock analysis tools, and more.

### Q: How can I contribute to CrewAI?
A: Contributions are welcome! You can fork the repository, create a new branch for your feature, add your improvement, and send a pull request. Check the Contribution section in the README for more details.
