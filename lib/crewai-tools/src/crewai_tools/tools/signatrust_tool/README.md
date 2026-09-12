# SignatrustTool Documentation

## Description

`SignatrustTool` lets your CrewAI agents generate, verify, and retrieve
**AI Decision Receipts** from [Signatrust](https://signatrust.net) — tamper-evident,
cryptographically signed (Ed25519) records of the decisions your agents make.

This is useful for **verifiable accountability and auditability** of AI-assisted
decisions (compliance reviews, approvals, financial actions, content moderation,
etc.). By default, only a SHA-256 hash of the decision payload is stored
server-side, so the tool is privacy-first.

The tool supports three operations:

- `generate` — create a new signed Decision Receipt for an agent decision
- `verify` — verify the cryptographic integrity of an existing receipt by id
- `get` — retrieve a stored receipt by id

## Installation

```shell
pip install 'crewai[tools]'
```

Then set your Signatrust API key:

```shell
export SIGNATRUST_API_KEY="sk_live_..."
```

Self-hosted deployments can point the tool at their own endpoint via `base_url`.

## Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import SignatrustTool

signatrust = SignatrustTool()  # reads SIGNATRUST_API_KEY from the environment

auditor = Agent(
    role="Compliance Auditor",
    goal="Record every approval decision as a verifiable Decision Receipt",
    backstory="You ensure all AI-assisted decisions are cryptographically logged.",
    tools=[signatrust],
    verbose=True,
)

task = Task(
    description=(
        "A loan application was approved by the model gpt-4o under the "
        "'KYC-2024' and 'AML-Tier1' policies, with human review. "
        "Generate a Signatrust Decision Receipt for this decision."
    ),
    expected_output="The receipt id and its verification status.",
    agent=auditor,
)

crew = Crew(agents=[auditor], tasks=[task])
crew.kickoff()
```

### Direct usage

```python
from crewai_tools import SignatrustTool

tool = SignatrustTool()

# Generate
print(tool.run(
    operation="generate",
    agent_name="loan-bot",
    action="approve_application",
    decision="approved",
    model="gpt-4o",
    policies=["KYC-2024", "AML-Tier1"],
    human_review=True,
))

# Verify
print(tool.run(operation="verify", receipt_id="rcpt_123"))

# Retrieve
print(tool.run(operation="get", receipt_id="rcpt_123"))
```

## Required environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SIGNATRUST_API_KEY` | Yes | Your Signatrust API key (`sk_live_...`). |

## Links

- Website: https://signatrust.net
- Source code: https://github.com/abokenan444/Signatrust
- Standalone package: `pip install crewai-signatrust` (also available)
