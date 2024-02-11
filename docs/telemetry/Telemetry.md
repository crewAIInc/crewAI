## Telemetry

CrewAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.

There is NO data being collected on the prompts, tasks descriptions agents backstories or goals nor tools usage, no API calls, nor responses nor any data that is being processed by the agents, nor any secrets and env vars.

Data collected includes:
- Version of crewAI
- Version of Python
- General OS (e.g. number of CPUs, macOS/Windows/Linux)
- Number of agents and tasks in a crew
- Crew Process being used
- If Agents are using memory or allowing delegation
- If Tasks are being executed in parallel or sequentially
- Language model being used
- Roles of agents in a crew
- Tools names available

Users can opt-in sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews.