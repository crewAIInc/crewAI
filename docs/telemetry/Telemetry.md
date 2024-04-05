---
title: Telemetry
description: Understanding the telemetry data collected by CrewAI and how it contributes to the enhancement of the library.
---
## Telemetry

CrewAI utilizes anonymous telemetry to gather usage statistics with the primary goal of enhancing the library. Our focus is on improving and developing the features, integrations, and tools most utilized by our users.

It's pivotal to understand that **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned. When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy.

### Data Collected Includes:
- **Version of CrewAI**: Assessing the adoption rate of our latest version helps us understand user needs and guide our updates.
- **Python Version**: Identifying the Python versions our users operate with assists in prioritizing our support efforts for these versions.
- **General OS Information**: Details like the number of CPUs and the operating system type (macOS, Windows, Linux) enable us to focus our development on the most used operating systems and explore the potential for OS-specific features.
- **Number of Agents and Tasks in a Crew**: Ensures our internal testing mirrors real-world scenarios, helping us guide users towards best practices.
- **Crew Process Utilization**: Understanding how crews are utilized aids in directing our development focus.
- **Memory and Delegation Use by Agents**: Insights into how these features are used help evaluate their effectiveness and future.
- **Task Execution Mode**: Knowing whether tasks are executed in parallel or sequentially influences our emphasis on enhancing parallel execution capabilities.
- **Language Model Utilization**: Supports our goal to improve support for the most popular languages among our users.
- **Roles of Agents within a Crew**: Understanding the various roles agents play aids in crafting better tools, integrations, and examples.
- **Tool Usage**: Identifying which tools are most frequently used allows us to prioritize improvements in those areas.

### Opt-In Further Telemetry Sharing
Users can choose to share their complete telemetry data by enabling the `share_crew` attribute to `True` in their crew configurations. This opt-in approach respects user privacy and aligns with data protection standards by ensuring users have control over their data sharing preferences. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

### Updates and Revisions
We are committed to maintaining the accuracy and transparency of our documentation. Regular reviews and updates are performed to ensure our documentation accurately reflects the latest developments of our codebase and telemetry practices. Users are encouraged to review this section for the most current information on our data collection practices and how they contribute to the improvement of CrewAI.