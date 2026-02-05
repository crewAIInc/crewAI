# Agent-OS Governance for CrewAI

Kernel-level policy enforcement for CrewAI workflows using [Agent-OS](https://github.com/imran-siddique/agent-os).

## Features

- **Policy Enforcement**: Define rules for agent behavior within crews
- **Tool Filtering**: Control which tools agents can use
- **Content Filtering**: Block dangerous patterns in outputs
- **Rate Limiting**: Limit iterations and tool calls
- **Audit Trail**: Full logging of all crew activities

## Installation

```bash
pip install crewai[governance]
# or
pip install agent-os-kernel
```

## Quick Start

```python
from crewai import Agent, Crew, Task
from crewai.governance import GovernedCrew, GovernancePolicy

# Create policy
policy = GovernancePolicy(
    max_tool_calls=20,
    max_iterations=15,
    blocked_patterns=["DROP TABLE", "rm -rf", "DELETE FROM"],
    blocked_tools=["shell_tool"],
)

# Create agents
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher",
)

writer = Agent(
    role="Writer",
    goal="Write clear reports",
    backstory="Technical writer",
)

# Create tasks
research_task = Task(
    description="Research AI governance",
    agent=researcher,
)

write_task = Task(
    description="Write summary report",
    agent=writer,
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
)

# Govern the crew
governed_crew = GovernedCrew(crew, policy)

# Execute with governance
result = governed_crew.kickoff()

# Check for violations
print(f"Violations: {len(governed_crew.violations)}")
print(f"Audit events: {len(governed_crew.audit_log)}")
```

## Policy Options

```python
GovernancePolicy(
    # Limits (tracked, enforcement requires CrewAI callbacks)
    max_tool_calls=50,       # Max tool invocations per task
    max_iterations=25,       # Max agent iterations
    max_execution_time=600,  # Max seconds for entire crew
    
    # Tool Control (enforced via tool filtering)
    allowed_tools=["search", "calculator"],  # Whitelist
    blocked_tools=["shell_tool", "file_delete"],  # Blacklist
    
    # Content Filtering (enforced on outputs)
    blocked_patterns=["DROP TABLE", "rm -rf"],
    max_output_length=100_000,
    
    # Audit
    log_all_actions=True,
)
```

## Handling Violations

```python
def on_violation(violation):
    print(f"BLOCKED: {violation.policy_name}")
    print(f"  Agent: {violation.agent_name}")
    print(f"  Reason: {violation.description}")
    # Send alert, log to SIEM, etc.

governed_crew = GovernedCrew(
    crew=crew,
    policy=policy,
    on_violation=on_violation,
)
```

## Audit Trail

```python
# Get detailed audit log
for event in governed_crew.audit_log:
    print(f"{event.timestamp}: {event.event_type}")
    if event.agent_name:
        print(f"  Agent: {event.agent_name}")
    print(f"  Details: {event.details}")

# Get summary
summary = governed_crew.get_audit_summary()
print(f"Total violations: {summary['total_violations']}")
print(f"Violations by type: {summary['violations_by_type']}")
```

## Integration with Agent-OS Kernel

For full kernel-level governance:

```python
from agent_os import KernelSpace
from agent_os.policies import SQLPolicy, CostControlPolicy
from crewai.governance import GovernedCrew

# Create kernel with policies
kernel = KernelSpace(policy=[
    SQLPolicy(allow=["SELECT"], deny=["DROP", "DELETE"]),
    CostControlPolicy(max_cost_usd=100),
])

# Wrap crew execution in kernel
@kernel.register
async def run_crew(inputs):
    return governed_crew.kickoff(inputs=inputs)

# Execute with full governance
result = await kernel.execute(run_crew, {"topic": "AI safety"})
```

## Links

- [Agent-OS GitHub](https://github.com/imran-siddique/agent-os)
- [CrewAI Documentation](https://docs.crewai.com)
