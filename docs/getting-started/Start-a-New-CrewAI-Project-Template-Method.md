---

title: Starting a New CrewAI Project - Using Template

description: A comprehensive guide to starting a new CrewAI project, including the latest updates and project setup methods.
---

# Starting Your CrewAI Project

Welcome to the ultimate guide for starting a new CrewAI project. This document will walk you through the steps to create, customize, and run your CrewAI project, ensuring you have everything you need to get started.

Before we start, there are a couple of things to note:

1. CrewAI is a Python package and requires Python >=3.10 and <=3.13 to run.
2. The preferred way of setting up CrewAI is using the `crewai create crew` command. This will create a new project folder and install a skeleton template for you to work on.

## Prerequisites

Before getting started with CrewAI, make sure that you have installed it via pip:

```shell
$ pip install 'crewai[tools]'
```

## Creating a New Project

In this example, we will be using poetry as our virtual environment manager.

To create a new CrewAI project, run the following CLI command:

```shell
$ crewai create crew <project_name>
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
  agent: researcher # THIS NEEDS TO MATCH THE AGENT NAME IN THE AGENTS.YAML FILE AND THE AGENT DEFINED IN THE crew.py FILE
  context: # THESE NEED TO MATCH THE TASK NAMES DEFINED ABOVE AND THE TASKS.YAML FILE AND THE TASK DEFINED IN THE crew.py FILE
    - researcher
```

### Referencing Variables:

Your defined functions with the same name will be used. For example, you can reference the agent for specific tasks from `tasks.yaml` file. Ensure your annotated agent and function name are the same; otherwise, your task won't recognize the reference properly.

#### Example References

`agents.yaml`

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

`tasks.yaml`

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

Use the annotations to properly reference the agent and task in the `crew.py` file.

### Annotations include:

* `@agent`
* `@task`
* `@crew`
* `@tool`
* `@callback`
* `@output_json`
* `@output_pydantic`
* `@cache_handler`

`crew.py`

```python
# ...
@agent
def email_summarizer(self) -> Agent:
    return Agent(
        config=self.agents_config["email_summarizer"],
    )

@task
def email_summarizer_task(self) -> Task:
    return Task(
        config=self.tasks_config["email_summarizer_task"],
    )
# ...
```

## Installing Dependencies

To install the dependencies for your project, you can use Poetry. First, navigate to your project directory:

```shell
$ cd my_project
$ crewai install
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

This will initialize your crew of AI agents and begin task execution as defined in your configuration in the `main.py` file.

### Replay Tasks from Latest Crew Kickoff

CrewAI now includes a replay feature that allows you to list the tasks from the last run and replay from a specific one. To use this feature, run:

```shell
$ crewai replay <task_id>
```

Replace `<task_id>` with the ID of the task you want to replay.

### Reset Crew Memory

If you need to reset the memory of your crew before running it again, you can do so by calling the reset memory feature:

```shell
$ crewai reset-memory
```

This will clear the crew's memory, allowing for a fresh start.

## Deploying Your Project

The easiest way to deploy your crew is through [CrewAI+](https://www.crewai.com/crewaiplus), where you can deploy your crew in a few clicks.
