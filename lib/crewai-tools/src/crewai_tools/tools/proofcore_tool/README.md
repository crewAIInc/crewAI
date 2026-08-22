# ProofCore Notarization Tool

This tool allows CrewAI agents to cryptographically notarize their generated outputs, audits, and agreements on the **TON Blockchain** using the Zero-Auth ProofCore Protocol.

## Installation

```bash
pip install 'crewai[tools]' requests
```

## Usage Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import ProofCoreTool

# 1. Initialize the ProofCore Notary Tool
notary_tool = ProofCoreTool()

# 2. Assign the tool to a compliance/auditor agent
auditor = Agent(
    role="Smart Contract Security Auditor",
    goal="Audit contract code and cryptographically seal the final verdict on-chain.",
    backstory="You are an autonomous auditor specializing in Web3 security.",
    tools=[notary_tool],
    verbose=True
)

audit_task = Task(
    description="Audit the token contract and notarize the report using ProofCore.",
    expected_output="A full security report concluding with the immutable ProofCore citation badge.",
    agent=auditor
)

crew = Crew(agents=[auditor], tasks=[audit_task])
result = crew.kickoff()
print(result)
```
