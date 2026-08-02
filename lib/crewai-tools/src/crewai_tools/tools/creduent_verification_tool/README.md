# Creduent Verification Tool

Verify external agent identities and cryptographic attestations locally before delegating tasks in CrewAI workflows.

## Installation

```bash
pip install creduent crewai-tools
```

## Usage

```python
from crewai_tools import CreduentVerificationTool

tool = CreduentVerificationTool()
result = tool.run(agent_uri="agent://assistant.dev/agent")
print(result)
```

## Protocol Specifications

Creduent performs local Ed25519 signature checks and canonical JSON (JCS RFC 8785) verification on agent URIs (`agent://<namespace>/<name>`) in under 5ms.
