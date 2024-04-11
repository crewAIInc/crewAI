---
title: Agent Monitoring with AgentOps
description: Understanding and logging your agent performance with AgentOps.
---

# Intro
Observability is a key aspect of developing and deploying conversational AI agents. It allows developers to understand how the agent is performing, how users are interacting with the agent, and how the agent is responding to user inputs. AgentOps is a product independent of CrewAI that provides a comprehensive observability solution for agents. 


## AgentOps

[AgentOps](https://agentops.ai) provides session replays, metrics, and monitoring for agents.
[AgentOps Repo](https://github.com/AgentOps-AI/agentops)

At a high level, AgentOps gives you the ability to monitor cost, token usage, latency, agent failures, session-wide statistics, and more.

### Overview
AgentOps provides monitoring for agents in development and production. It provides a dashboard for monitoring agent performance, session replays, and custom reporting.

Additionally, AgentOps provides session drilldowns that allows users to view the agent's interactions with users in real-time. This feature is useful for debugging and understanding how the agent interacts with users.

![Agent Sessions Overview](..%2Fassets%2Fagentops-overview.png)
![Session Drilldowns](..%2Fassets%2Fagentops-session.png)
![Agent Replays](..%2Fassets%2Fagentops-replay.png)

### Features
- LLM Cost Management and Tracking
- Replay Analytics
- Recursive Thought Detection
- Custom Reporting
- Analytics Dashboard
- Public Model Testing
- Custom Tests
- Time Travel Debugging
- Compliance and Security
- Prompt Injection Detection

### Using AgentOps

1. **Create an API Key:**
Create a user API key here: [Create API Key](app.agentops.ai/account)

2. **Configure Your Environment:**
Add your API key to your environment variables

```
AGENTOPS_API_KEY=<YOUR_AGENTOPS_API_KEY>
```

3. **Install AgentOps:**
Install AgentOps with:
```
pip install crewai[agentops]
```
or
```
pip install agentops
```

Before using `Crew` in your script, include these lines:

```python
import agentops
agentops.init()
```

This will initiate an AgentOps session as well as automatically track Crew agents. For futher info on how to outfit more complex agentic systems, check out the [AgentOps documentation](https://docs.agentops.ai) or join the [Discord](https://discord.gg/j4f3KbeH).

### Crew + AgentOps Examples
- [Job Posting](https://github.com/joaomdmoura/crewAI-examples/tree/main/job-posting)
- [Markdown Validator](https://github.com/joaomdmoura/crewAI-examples/tree/main/markdown_validator)
- [Instagram Post](https://github.com/joaomdmoura/crewAI-examples/tree/main/instagram_post)


### Further Information
To implement more features and better observability, please see the [AgentOps Repo](https://github.com/AgentOps-AI/agentops).
