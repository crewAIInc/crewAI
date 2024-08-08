---
title: Starting a New CrewAI Project - Using Template
description: A comprehensive guide to starting a new CrewAI project, including the latest updates and project setup methods.
---

# Starting Your CrewAI Project

Welcome to the ultimate guide for starting a new CrewAI project. This document will walk you through the steps to create, customize, and run your CrewAI project, ensuring you have everything you need to get started.

Beforre we start there are a couple of things to note:

1. CrewAI is a Python package and requires Python >=3.10 and <=3.13 to run.
2. The preferred way of setting up CrewAI is using the `crewai create` command.This will create a new project folder and install a skeleton template for you to work on.

## Prerequisites

Before getting started with CrewAI, make sure that you have installed it via pip:

```shell
$ pip install crewai crewai-tools
```

### Virtual Environments
It is highly recommended that you use virtual environments to ensure that your CrewAI project is isolated from other projects and dependencies. Virtual environments provide a clean, separate workspace for each project, preventing conflicts between different versions of packages and libraries. This isolation is crucial for maintaining consistency and reproducibility in your development process. You have multiple options for setting up virtual environments depending on your operating system and Python version:

1. Use venv (Python's built-in virtual environment tool):
   venv is included with Python 3.3 and later, making it a convenient choice for many developers. It's lightweight and easy to use, perfect for simple project setups.

   To set up virtual environments with venv, refer to the official [Python documentation](https://docs.python.org/3/tutorial/venv.html).

2. Use Conda (A Python virtual environment manager):
   Conda is an open-source package manager and environment management system for Python. It's widely used by data scientists, developers, and researchers to manage dependencies and environments in a reproducible way.

   To set up virtual environments with Conda, refer to the official [Conda documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

3. Use Poetry (A Python package manager and dependency management tool):
   Poetry is an open-source Python package manager that simplifies the installation of packages and their dependencies. Poetry offers a convenient way to manage virtual environments and dependencies.
   Poetry is CrewAI's prefered tool for package / dependancy management in CrewAI.

### Code IDEs

Most users of CrewAI a Code Editor / Integrated Development Environment (IDE) for building there Crews. You can use any code IDE of your choice. Seee below for some popular options for Code Editors / Integrated Development Environments (IDE):

- [Visual Studio Code](https://code.visualstudio.com/) - Most popular
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Cursor AI](https://cursor.com)

Pick one that suits your style and needs.

## Creating a New Project
In this example we will be using Venv as our virtual environment manager.

To setup a virtual environment, run the following CLI command:

```shell
$ python3 -m venv <venv-name>
```

Activate your virtual environment by running the following CLI command:

```shell
$ source <venv-name>/bin/activate
```

Now, to create a new CrewAI project, run the following CLI command:

```shell
$ crewai create <project_name>
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
  agent: researcher # THIS NEEDS TO MATCH THE AGENT NAME IN THE AGENTS.YAML FILE AND THE AGENT DEFINED IN THE Crew.PY FILE
  context: # THESE NEED TO MATCH THE TASK NAMES DEFINED ABOVE AND THE TASKS.YAML FILE AND THE TASK DEFINED IN THE Crew.PY FILE
    - researcher
```

### Referencing Variables:
Your defined functions with the same name will be used. For example, you can reference the agent for specific tasks from task.yaml file. Ensure your annotated agent and function name is the same otherwise your task wont recognize the reference properly.

#### Example References
agent.yaml
```yaml
email_summarizer:
    role: >
      Email Summarizer
    goal: >
      Summarize emails into a concise and clear summary
    backstory: >
      You will create a 5 bullet point summary of the report
    llm: mixtal_llm
```

task.yaml
```yaml
email_summarizer_task:
    description: >
      Summarize the email into a 5 bullet point summary
    expected_output: >
      A 5 bullet point summary of the email
    agent: email_summarizer
    context:
      - reporting_task
      - research_task
```

Use the annotations are used to properly reference the agent and task in the crew.py file.

### Annotations include:
* @agent
* @task
* @crew
* @llm
* @tool
* @callback
* @output_json
* @output_pydantic
* @cache_handler


crew.py
```py
...
    @llm
    def mixtal_llm(self):
        return ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    @agent
    def email_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["email_summarizer"],
        )
    ## ...other tasks defined
    @task
    def email_summarizer_task(self) -> Task:
        return Task(
            config=self.tasks_config["email_summarizer_task"],
        )
...
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
$ crewai run
```
or
```shell
$ poetry run my_project
```

This will initialize your crew of AI agents and begin task execution as defined in your configuration in the `main.py` file.

## Deploying Your Project

The easiest way to deploy your crew is through [CrewAI+](https://www.crewai.com/crewaiplus), where you can deploy your crew in a few clicks.
