---
title: Starting a New CrewAI Project
description: A comprehensive guide to starting a new CrewAI project, including the latest updates and project setup methods.
---

# Starting Your CrewAI Project

Welcome to the ultimate guide for starting a new CrewAI project. This document will walk you through the steps to create, customize, and run your CrewAI project, ensuring you have everything you need to get started.

## Prerequisites

We assume you have already installed CrewAI. If not, please refer to the [installation guide](how-to/Installing-CrewAI.md) to install CrewAI and its dependencies.

## Creating a New Project

To create a new project, run the following CLI command:

```shell
$ crewai create my_project
```

This command will create a new project folder with the following structure:

```shell
my_project/
├── .gitignore
├── pyproject.toml
├── README.md
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

You can now start developing your project by editing the files in the `src/my_project` folder. The `main.py` file is the entry point of your project, and the `crew.py` file is where you define your agents and tasks.

## Customizing Your Project

To customize your project, you can:
- Modify `src/my_project/config/agents.yaml` to define your agents.
- Modify `src/my_project/config/tasks.yaml` to define your tasks.
- Modify `src/my_project/crew.py` to add your own logic, tools, and specific arguments.
- Modify `src/my_project/main.py` to add custom inputs for your agents and tasks.
- Add your environment variables into the `.env` file.

### Example: Defining Agents and Tasks

#### agents.yaml

```yaml
researcher:
  role: >
    Job Candidate Researcher
  goal: >
    Find potential candidates for the job
  backstory: >
    You are adept at finding the right candidates by exploring various online
    resources. Your skill in identifying suitable candidates ensures the best
    match for job positions.
```

#### tasks.yaml

```yaml
research_candidates_task:
  description: >
    Conduct thorough research to find potential candidates for the specified job.
    Utilize various online resources and databases to gather a comprehensive list of potential candidates.
    Ensure that the candidates meet the job requirements provided.

    Job Requirements:
    {job_requirements}
  expected_output: >
    A list of 10 potential candidates with their contact information and brief profiles highlighting their suitability.
```

## Installing Dependencies

To install the dependencies for your project, you can use Poetry. First, navigate to your project directory:

```shell
$ cd my_project
$ poetry lock
$ poetry install
```

This will install the dependencies specified in the `pyproject.toml` file.

## Interpolating Variables

Any variable interpolated in your `agents.yaml` and `tasks.yaml` files like `{variable}` will be replaced by the value of the variable in the `main.py` file.

#### agents.yaml

```yaml
research_task:
  description: >
    Conduct a thorough research about the customer and competitors in the context
    of {customer_domain}.
    Make sure you find any interesting and relevant information given the
    current year is 2024.
  expected_output: >
    A complete report on the customer and their customers and competitors,
    including their demographics, preferences, market positioning and audience engagement.
```

#### main.py

```python
# main.py
def run():
    inputs = {
        "customer_domain": "crewai.com"
    }
    MyProjectCrew(inputs).crew().kickoff(inputs=inputs)
```

## Running Your Project

To run your project, use the following command:

```shell
$ poetry run my_project
```

This will initialize your crew of AI agents and begin task execution as defined in your configuration in the `main.py` file.

## Deploying Your Project

The easiest way to deploy your crew is through [CrewAI+](https://www.crewai.com/crewaiplus), where you can deploy your crew in a few clicks.