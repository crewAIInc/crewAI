---
title: CrewAI Agent Monitoring with Langtrace
description: How to monitor cost, latency, and performance of CrewAI Agents using Langtrace, an external observability tool.
---

# Langtrace Overview

Langtrace is an open-source, external tool that helps you set up observability and evaluations for Large Language Models (LLMs), LLM frameworks, and Vector Databases. While not built directly into CrewAI, Langtrace can be used alongside CrewAI to gain deep visibility into the cost, latency, and performance of your CrewAI Agents. This integration allows you to log hyperparameters, monitor performance regressions, and establish a process for continuous improvement of your Agents.

## Setup Instructions

1. Sign up for [Langtrace](https://langtrace.ai/) by visiting [https://langtrace.ai/signup](https://langtrace.ai/signup).
2. Create a project and generate an API key.
3. Install Langtrace in your CrewAI project using the following commands:

```bash
# Install the SDK
pip install langtrace-python-sdk
```

## Using Langtrace with CrewAI

To integrate Langtrace with your CrewAI project, follow these steps:

1. Import and initialize Langtrace at the beginning of your script, before any CrewAI imports:

```python
from langtrace_python_sdk import langtrace
langtrace.init(api_key='<LANGTRACE_API_KEY>')

# Now import CrewAI modules
from crewai import Agent, Task, Crew
```

2. Create your CrewAI agents and tasks as usual.

3. Use Langtrace's tracking functions to monitor your CrewAI operations. For example:

```python
with langtrace.trace("CrewAI Task Execution"):
    result = crew.kickoff()
```

### Features and Their Application to CrewAI

1. **LLM Token and Cost Tracking**
   - Monitor the token usage and associated costs for each CrewAI agent interaction.
   - Example:
     ```python
     with langtrace.trace("Agent Interaction"):
         agent_response = agent.execute(task)
     ```

2. **Trace Graph for Execution Steps**
   - Visualize the execution flow of your CrewAI tasks, including latency and logs.
   - Useful for identifying bottlenecks in your agent workflows.

3. **Dataset Curation with Manual Annotation**
   - Create datasets from your CrewAI task outputs for future training or evaluation.
   - Example:
     ```python
     langtrace.log_dataset_item(task_input, agent_output, {"task_type": "research"})
     ```

4. **Prompt Versioning and Management**
   - Keep track of different versions of prompts used in your CrewAI agents.
   - Useful for A/B testing and optimizing agent performance.

5. **Prompt Playground with Model Comparisons**
   - Test and compare different prompts and models for your CrewAI agents before deployment.

6. **Testing and Evaluations**
   - Set up automated tests for your CrewAI agents and tasks.
   - Example:
     ```python
     langtrace.evaluate(agent_output, expected_output, "accuracy")
     ```

## Monitoring New CrewAI Features

CrewAI has introduced several new features that can be monitored using Langtrace:

1. **Code Execution**: Monitor the performance and output of code executed by agents.
   ```python
   with langtrace.trace("Agent Code Execution"):
       code_output = agent.execute_code(code_snippet)
   ```

2. **Third-party Agent Integration**: Track interactions with LlamaIndex, LangChain, and Autogen agents.