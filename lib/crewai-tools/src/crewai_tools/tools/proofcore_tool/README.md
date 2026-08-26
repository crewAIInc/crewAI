# ProofCore Notarization & Verification Tools

This package allows CrewAI agents to cryptographically seal their generated outputs, audits, and agreements on the **TON Blockchain** using the Zero-Auth ProofCore Protocol, as well as programmatically verify attestations from other agents.

## Installation

```bash
pip install 'crewai[tools]' requests
```

## Available Tools

1. **`ProofCoreTool`**: Cryptographically commits text/audits to the TON Blockchain with an instant Ed25519 notary signature.
2. **`ProofCoreVerifyTool`**: Programmatically verifies the authenticity of a sealed deal, confirming the content hash, Ed25519 signature, and on-chain TON Merkle anchor.

---

## Usage Examples

### 1. Notarizing Findings on Blockchain (`ProofCoreTool`)

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

---

### 2. Verifying Counterparty Claims (`ProofCoreVerifyTool`)

```python
from crewai import Agent, Task, Crew
from crewai_tools import ProofCoreVerifyTool

# 1. Initialize the Verifier Tool
verifier_tool = ProofCoreVerifyTool()

# 2. Assign the tool to a verification agent
verifier = Agent(
    role="Compliance Verifier",
    goal="Verify that incoming reports and audits are genuine and un-tampered.",
    backstory="You are an autonomous verification agent auditing data provenance.",
    tools=[verifier_tool],
    verbose=True
)

verify_task = Task(
    description=(
        "Verify the authenticity of deal '4eea9784-2371-4505-8f2f-0b4c5a15a9ec' "
        "with content 'AUDIT REPORT: Vault.sol...' using the ProofCore verifier tool."
    ),
    expected_output="A status report confirming if the signature and blockchain anchor are valid.",
    agent=verifier
)

crew = Crew(agents=[verifier], tasks=[verify_task])
result = crew.kickoff()
print(result)
```
