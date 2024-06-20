---
title: Initial Support to Bring Your Own Prompts in CrewAI
description: Enhancing customization and internationalization by allowing users to bring their own prompts in CrewAI.

---

# Initial Support to Bring Your Own Prompts in CrewAI

CrewAI now supports the ability to bring your own prompts, enabling extensive customization and internationalization. This feature allows users to tailor the inner workings of their agents to better suit specific needs, including support for multiple languages.

## Internationalization and Customization Support

### Custom Prompts with `prompt_file`

The `prompt_file` attribute facilitates full customization of the agent prompts, enhancing the global usability of CrewAI. Users can specify their prompt templates, ensuring that the agents communicate in a manner that aligns with specific project requirements or language preferences.

#### Example of a Custom Prompt File

The custom prompts can be defined in a JSON file, similar to the example provided [here](https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/translations/en.json).

### Supported Languages

CrewAI's custom prompt support includes internationalization, allowing prompts to be written in different languages. This is particularly useful for global teams or projects that require multilingual support.

## How to Use the `prompt_file` Attribute

To utilize the `prompt_file` attribute, include it in your crew definition. Below is an example demonstrating how to set up agents and tasks with custom prompts.

### Example

```python
import os
from crewai import Agent, Task, Crew

# Define your agents
researcher = Agent(
    role="Researcher",
    goal="Make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Write the best content about AI and AI agents.",
    backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
    allow_delegation=False,
)

# Define your tasks
tasks = [
    Task(
        description="Say Hi",
        expected_output="The word: Hi",
        agent=researcher,
    )
]

# Instantiate your crew with custom prompts
crew = Crew(
    agents=[researcher],
    tasks=tasks,
    prompt_file="prompt.json",  # Path to your custom prompt file
)

# Get your crew to work!
crew.kickoff()
```

## Advanced Customization Features

### `language` Attribute

In addition to `prompt_file`, the `language` attribute can be used to specify the language for the agent's prompts. This ensures that the prompts are generated in the desired language, further enhancing the internationalization capabilities of CrewAI.

### Creating Custom Prompt Files

Custom prompt files should be structured in JSON format and include all necessary prompt templates. Below is a simplified example of a prompt JSON file:

```json
{
    "system": "You are a system template.",
    "prompt": "Here is your prompt template.",
    "response": "Here is your response template."
}
```

### Benefits of Custom Prompts

- **Enhanced Flexibility**: Tailor agent communication to specific project needs.
- **Improved Usability**: Supports multiple languages, making it suitable for global projects.
- **Consistency**: Ensures uniform prompt structures across different agents and tasks.

By incorporating these updates, CrewAI provides users with the ability to fully customize and internationalize their agent prompts, making the platform more versatile and user-friendly.
