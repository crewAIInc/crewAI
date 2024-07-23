---
title: Replay Tasks from Latest Crew Kickoff
description: Replay tasks from the latest crew.kickoff(...)
---

## Introduction
CrewAI provides the ability to replay from a task specified from the latest crew kickoff. This feature is particularly useful when you've finished a kickoff and may want to retry certain tasks or don't need to refetch data over and your agents already have the context saved from the kickoff execution so you just need to replay the tasks you want to.

## Note:
You must run `crew.kickoff()` before you can replay a task. Currently, only the latest kickoff is supported, so if you use `kickoff_for_each`, it will only allow you to replay from the most recent crew run.

Here's an example of how to replay from a task:

### Replaying from specific task Using the CLI
To use the replay feature, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where your CrewAI project is located.
3. Run the following command:

To view latest kickoff task_ids use:
```shell
crewai log-tasks-outputs
```

Once you have your task_id to replay from use:
```shell
crewai replay -t <task_id>
```


### Replaying from a task Programmatically
To replay from a task programmatically, use the following steps:

1. Specify the task_id and input parameters for the replay process.
2. Execute the replay command within a try-except block to handle potential errors.

```python
   def replay():
    """
    Replay the crew execution from a specific task.
    """
    task_id = '<task_id>'
    inputs = {"topic": "CrewAI Training"} # this is optional, you can pass in the inputs you want to replay otherwise uses the previous kickoffs inputs
    try:
        YourCrewName_Crew().crew().replay(task_id=task_id, inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")