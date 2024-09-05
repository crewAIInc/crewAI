---
title: CrewAI Agent Monitoring with Langtrace
description: How to monitor cost, latency, and performance of CrewAI Agents using Langtrace, an external observability tool.
---

# Langtrace Overview

Langtrace is an open-source, external tool that helps you set up observability and evaluations for Large Language Models (LLMs), LLM frameworks, and Vector Databases. While not built directly into CrewAI, Langtrace can be used alongside CrewAI to gain deep visibility into the cost, latency, and performance of your CrewAI Agents. This integration allows you to log hyperparameters, monitor performance regressions, and establish a process for continuous improvement of your Agents.

![Overview of a select series of agent session runs](..%2Fassets%2Flangtrace1.png)
![Overview of agent traces](..%2Fassets%2Flangtrace2.png)
![Overview of llm traces in details](..%2Fassets%2Flangtrace3.png)

## Setup Instructions

1. Sign up for [Langtrace](https://langtrace.ai/) by visiting [https://langtrace.ai/signup](https://langtrace.ai/signup).
2. Create a project, set the project type to crewAI & generate an API key.
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

### Features and Their Application to CrewAI

1. **LLM Token and Cost Tracking**

   - Monitor the token usage and associated costs for each CrewAI agent interaction.

2. **Trace Graph for Execution Steps**

   - Visualize the execution flow of your CrewAI tasks, including latency and logs.
   - Useful for identifying bottlenecks in your agent workflows.

3. **Dataset Curation with Manual Annotation**

   - Create datasets from your CrewAI task outputs for future training or evaluation.

4. **Prompt Versioning and Management**

   - Keep track of different versions of prompts used in your CrewAI agents.
   - Useful for A/B testing and optimizing agent performance.

5. **Prompt Playground with Model Comparisons**

   - Test and compare different prompts and models for your CrewAI agents before deployment.

6. **Testing and Evaluations**
   - Set up automated tests for your CrewAI agents and tasks.
