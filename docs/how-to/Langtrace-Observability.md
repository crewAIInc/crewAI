---
title: CrewAI Agent Monitoring with Langtrace
description: How to monitor cost, latency and performance of CrewAI Agents using Langtrace.
---

# Langtrace Overview
Langtrace is an open-source tool that helps you set up observability and evaluations for LLMs, LLM frameworks and VectorDB. With Langtrace, you can get deep visibility into the cost, latency and performance of your CrewAI Agents. Additionally, you can log the hyperparameters and monitor for any performance regressions and set up a process to continuously improve your Agents. 

## Setup Instructions

1. Sign up for [Langtrace](https://langtrace.ai/) by going to [https://langtrace.ai/signup](https://langtrace.ai/signup). 
2. Create a project and generate an API key. 
3. Install Langtrace in your code using the following commands.
**Note**: For detailed instructions on integrating Langtrace, you can check out the official docs from [here](https://docs.langtrace.ai/supported-integrations/llm-frameworks/crewai).

```
# Install the SDK
pip install langtrace-python-sdk

# Import it into your project
from langtrace_python_sdk import langtrace # Must precede any llm module imports
langtrace.init(api_key = '<LANGTRACE_API_KEY>')
```

### Features
- **LLM Token and Cost tracking**
- **Trace graph showing detailed execution steps with latency and logs**
- **Dataset curation using manual annotation**
- **Prompt versioning and management**
- **Prompt Playground with comparison views between models**
- **Testing and Evaluations**

![Langtrace Cost and Usage Tracking](..%2Fassets%2Fcrewai-langtrace-stats.png)
![Langtrace Span Graph and Logs Dashboard](..%2Fassets%2Fcrewai-langtrace-spans.png)

#### Extra links

<a href="https://x.com/langtrace_ai">🐦 Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://discord.com/invite/EaSATwtr4t">📢 Discord</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://langtrace.ai/">🖇 Website</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://docs.langtrace.ai/introduction">📙 Documentation</a>
