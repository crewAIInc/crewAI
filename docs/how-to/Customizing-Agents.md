---
title: Customizing Agents in CrewAI
description: A comprehensive guide to tailoring agents for specific roles, tasks, and advanced customizations within the CrewAI framework.
---

## Customizable Attributes
Crafting an efficient CrewAI team hinges on the ability to dynamically tailor your AI agents to meet the unique requirements of any project. This section covers the foundational attributes you can customize.

### Key Attributes for Customization
- **Role**: Specifies the agent's job within the crew, such as 'Analyst' or 'Customer Service Rep'.
- **Goal**: Defines what the agent aims to achieve, in alignment with its role and the overarching objectives of the crew.
- **Backstory**: Provides depth to the agent's persona, enriching its motivations and engagements within the crew.
- **Tools**: Represents the capabilities or methods the agent uses to perform tasks, from simple functions to intricate integrations.

## Advanced Customization Options
Beyond the basic attributes, CrewAI allows for deeper customization to enhance an agent's behavior and capabilities significantly.

### Language Model Customization
Agents can be customized with specific language models (`llm`) and function-calling language models (`function_calling_llm`), offering advanced control over their processing and decision-making abilities. It's important to note that setting the `function_calling_llm` allows for overriding the default crew function-calling language model, providing a greater degree of customization.

## Performance and Debugging Settings
Adjusting an agent's performance and monitoring its operations are crucial for efficient task execution.

### Verbose Mode and RPM Limit
- **Verbose Mode**: Enables detailed logging of an agent's actions, useful for debugging and optimization. Specifically, it provides insights into agent execution processes, aiding in the optimization of performance.
- **RPM Limit**: Sets the maximum number of requests per minute (`max_rpm`). This attribute is optional and can be set to `None` for no limit, allowing for unlimited queries to external services if needed.

### Maximum Iterations for Task Execution
The `max_iter` attribute allows users to define the maximum number of iterations an agent can perform for a single task, preventing infinite loops or excessively long executions. The default value is set to 15, providing a balance between thoroughness and efficiency. Once the agent approaches this number, it will try its best to give a good answer.

## Customizing Agents and Tools
Agents are customized by defining their attributes and tools during initialization. Tools are critical for an agent's functionality, enabling them to perform specialized tasks. The `tools` attribute should be an array of tools the agent can utilize, and it's initialized as an empty list by default. Tools can be added or modified post-agent initialization to adapt to new requirements.

```shell
pip install 'crewai[tools]'
```

### Example: Assigning Tools to an Agent
```python
import os
from crewai import Agent
from crewai_tools import SerperDevTool

# Set API keys for tool initialization
os.environ["OPENAI_API_KEY"] = "Your Key"
os.environ["SERPER_API_KEY"] = "Your Key"

# Initialize a search tool
search_tool = SerperDevTool()

# Initialize the agent with advanced options
agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[search_tool],
  memory=True, # Enable memory
  verbose=True,
  max_rpm=None, # No limit on requests per minute
  max_iter=15, # Default value for maximum iterations
  allow_delegation=False
)
```

## Delegation and Autonomy
Controlling an agent's ability to delegate tasks or ask questions is vital for tailoring its autonomy and collaborative dynamics within the CrewAI framework. By default, the `allow_delegation` attribute is set to `True`, enabling agents to seek assistance or delegate tasks as needed. This default behavior promotes collaborative problem-solving and efficiency within the CrewAI ecosystem. If needed, delegation can be disabled to suit specific operational requirements.

### Example: Disabling Delegation for an Agent
```python
agent = Agent(
  role='Content Writer',
  goal='Write engaging content on market trends',
  backstory='A seasoned writer with expertise in market analysis.',
  allow_delegation=False # Disabling delegation
)
```

## Conclusion
Customizing agents in CrewAI by setting their roles, goals, backstories, and tools, alongside advanced options like language model customization, memory, performance settings, and delegation preferences, equips a nuanced and capable AI team ready for complex challenges.