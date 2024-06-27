---
title: Agent Monitoring with AgentOps
description: Understanding and logging your agent performance with AgentOps.
---

# Intro
Observability is a key aspect of developing and deploying conversational AI agents. It allows developers to understand how their agents are performing, how their agents are interacting with users, and how their agents use external tools and APIs. AgentOps is a product independent of CrewAI that provides a comprehensive observability solution for agents.

## AgentOps

[AgentOps](https://agentops.ai/?=crew) provides session replays, metrics, and monitoring for agents.

At a high level, AgentOps gives you the ability to monitor cost, token usage, latency, agent failures, session-wide statistics, and more. For more info, check out the [AgentOps Repo](https://github.com/AgentOps-AI/agentops).

### Overview
AgentOps provides monitoring for agents in development and production. It provides a dashboard for tracking agent performance, session replays, and custom reporting.

Additionally, AgentOps provides session drilldowns for viewing Crew agent interactions, LLM calls, and tool usage in real-time. This feature is useful for debugging and understanding how agents interact with users as well as other agents.

![Overview of a select series of agent session runs](..%2Fassets%2Fagentops-overview.png)
![Overview of session drilldowns for examining agent runs](..%2Fassets%2Fagentops-session.png)
![Viewing a step-by-step agent replay execution graph](..%2Fassets%2Fagentops-replay.png)

### Features
- **LLM Cost Management and Tracking**: Track spend with foundation model providers.
- **Replay Analytics**: Watch step-by-step agent execution graphs.
- **Recursive Thought Detection**: Identify when agents fall into infinite loops.
- **Custom Reporting**: Create custom analytics on agent performance.
- **Analytics Dashboard**: Monitor high-level statistics about agents in development and production.
- **Public Model Testing**: Test your agents against benchmarks and leaderboards.
- **Custom Tests**: Run your agents against domain-specific tests.
- **Time Travel Debugging**: Restart your sessions from checkpoints.
- **Compliance and Security**: Create audit logs and detect potential threats such as profanity and PII leaks.
- **Prompt Injection Detection**: Identify potential code injection and secret leaks.

### Using AgentOps

1. **Create an API Key:**
   Create a user API key here: [Create API Key](app.agentops.ai/account)

2. **Configure Your Environment:**
   Add your API key to your environment variables

   ```bash
   AGENTOPS_API_KEY=<YOUR_AGENTOPS_API_KEY>
   ```

3. **Install AgentOps:**
   Install AgentOps with:
   ```bash
   pip install crewai[agentops]
   ```
   or
   ```bash
   pip install agentops
   ```

   Before using `Crew` in your script, include these lines:

   ```python
   import agentops
   agentops.init()
   ```

   This will initiate an AgentOps session as well as automatically track Crew agents. For further info on how to outfit more complex agentic systems, check out the [AgentOps documentation](https://docs.agentops.ai) or join the [Discord](https://discord.gg/j4f3KbeH).

### Crew + AgentOps Examples
- [Job Posting](https://github.com/joaomdmoura/crewAI-examples/tree/main/job-posting)
- [Markdown Validator](https://github.com/joaomdmoura/crewAI-examples/tree/main/markdown_validator)
- [Instagram Post](https://github.com/joaomdmoura/crewAI-examples/tree/main/instagram_post)

### Further Information

To get started, create an [AgentOps account](https://agentops.ai/?=crew).

For feature requests or bug reports, please reach out to the AgentOps team on the [AgentOps Repo](https://github.com/AgentOps-AI/agentops).

#### Extra links

<a href="https://twitter.com/agentopsai/">🐦 Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://discord.gg/JHPt4C7r">📢 Discord</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://app.agentops.ai/?=crew">🖇️ AgentOps Dashboard</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://docs.agentops.ai/introduction">📙 Documentation</a>