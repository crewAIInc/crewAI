---
title: Forcing Tool Output as Result
description: Learn how to force tool output as the result in of an Agent's task in crewAI.
---

## Introduction
In CrewAI, you can force the output of a tool as the result of an agent's task. This feature is useful when you want to ensure that the tool output is captured and returned as the task result, and avoid the agent modifying the output during the task execution.

## Forcing Tool Output as Result
To force the tool output as the result of an agent's task, you can set the `result_as_answer` parameter to `True` when creating the agent. This parameter ensures that the tool output is captured and returned as the task result, without any modifications by the agent.

Here's an example of how to force the tool output as the result of an agent's task:

```python
# ...
# Define a custom tool that returns the result as the answer
coding_agent =Agent(
        role="Data Scientist",
        goal="Product amazing reports on AI",
        backstory="You work with data and AI",
        tools=[MyCustomTool(result_as_answer=True)],
    )
# ...
```

### Workflow in Action

1. **Task Execution**: The agent executes the task using the tool provided.
2. **Tool Output**: The tool generates the output, which is captured as the task result.
3. **Agent Interaction**: The agent my reflect and take learnings from the tool but the output is not modified.
4. **Result Return**: The tool output is returned as the task result without any modifications.
