# ComposioTool Documentation

## Description

This tools is a wrapper around the composio toolset and gives your agent access to a wide variety of tools from the composio SDK.

## Installation

To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install composio-core 
pip install 'crewai[tools]'
```

after the installation is complete, either run `composio login` or export your composio API key as `COMPOSIO_API_KEY`.

## Example

The following example demonstrates how to initialize the tool and execute a github action:

1. Initialize toolset

```python
from composio import App
from crewai_tools import ComposioTool
from crewai import Agent, Task


tools = [ComposioTool.from_action(action=Action.GITHUB_ACTIVITY_STAR_REPO_FOR_AUTHENTICATED_USER)]
```

If you don't know what action you want to use, use `from_app` and `tags` filter to get relevant actions

```python
tools = ComposioTool.from_app(App.GITHUB, tags=["important"])
```

or use `use_case` to search relevant actions

```python
tools = ComposioTool.from_app(App.GITHUB, use_case="Star a github repository")
```

2. Define agent

```python
crewai_agent = Agent(
    role="Github Agent",
    goal="You take action on Github using Github APIs",
    backstory=(
        "You are AI agent that is responsible for taking actions on Github "
        "on users behalf. You need to take action on Github using Github APIs"
    ),
    verbose=True,
    tools=tools,
)
```

3. Execute task

```python
task = Task(
    description="Star a repo ComposioHQ/composio on GitHub",
    agent=crewai_agent,
    expected_output="if the star happened",
)

task.execute()
```

* More detailed list of tools can be found [here](https://app.composio.dev)
