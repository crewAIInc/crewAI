# CCS Security Integration for CrewAI

[CCS](https://github.com/Correctover/ccs-verifier) provides sub-millisecond runtime security verification for CrewAI agent tool execution.

## Quick Start

```python
from crewai import Agent, Task, Crew
from examples.ccs_security.ccs_guard import CCSSecurityMiddleware

# Wrap tools with CCS security
secured_tools = [CCSSecurityMiddleware.wrap_tool(t) for t in agent_tools]

agent = Agent(
    role="Research Assistant",
    tools=secured_tools,
    ...
)
```

## How It Works

CCS verifies every tool call in-process (~7.5μs P50) before execution:
- RCE attempts → blocked
- SSRF requests → blocked  
- Credential leaks → blocked
- Safe operations → pass through instantly

## Install

```bash
pip install ccs-verifier
```

## Reference

- [CCS IETF Draft](https://datatracker.ietf.org/doc/draft-correctover-ccs/)
- [CCS Zenodo DOI](https://doi.org/10.5281/zenodo.21783723)
