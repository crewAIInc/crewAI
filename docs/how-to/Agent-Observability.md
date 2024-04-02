---
title: Agent Observability
description: Understanding and logging your agent performance with AgentOps.
---

# AgentOps

[AgentOps](https://agentops.ai) provides session replays, metrics, and monitoring for agents built with Superagent.
[AgentOps Repo](https://github.com/AgentOps-AI/agentops)

## Overview
AgentOps provides monotoring for agents in development and production. It provides a dashboard for monitoring agent performance, session replays, and custom reporting.

![agentops-overview.png](..%2Fassets%2Fagentops-overview.png)

Additionally, AgentOps provides session drilldowns that allows users to view the agent's interactions with users in real-time. This feature is useful for debugging and understanding how the agent interacts with users.

![agentops-session.png](..%2Fassets%2Fagentops-session.png)
![agentops-replay.png](..%2Fassets%2Fagentops-replay.png)

## Features
- LLM Cost management and tracking
- Replay Analytics
- Recursive thought detection
- Custom Reporting
- Analytics Dashboard
- Public Model Testing
- Custom Tests
- Time Travel Debugging
- Compliance and Security

## Using AgentOps

Create a user API key here: app.agentops.ai/account

Add your API key to your environment variables

```
AGENTOPS_API_KEY=<YOUR_AGENTOPS_API_KEY>
```

Before using `Crew` in your script, include these lines:

```python
import agentops
agentops.init()
```

### Crew + AgentOps Examples
- [Job Posting](https://github.com/joaomdmoura/crewAI-examples/tree/main/job-posting)
- [Markdown Validator](https://github.com/joaomdmoura/crewAI-examples/tree/main/markdown_validator)
- [Instagram Post](https://github.com/joaomdmoura/crewAI-examples/tree/main/instagram_post)


### Futher Information
To implement more features and better observability, please see the [AgentOps Repo](https://github.com/AgentOps-AI/agentops)
