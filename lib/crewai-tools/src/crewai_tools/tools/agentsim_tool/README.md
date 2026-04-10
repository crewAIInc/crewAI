# AgentSIM Tool

Real carrier-grade phone number provisioning and OTP verification for CrewAI agents.

[AgentSIM](https://agentsim.dev) provides real T-Mobile SIM numbers that pass carrier lookup checks as `line_type: mobile`. Unlike VoIP numbers (Twilio, Google Voice), these numbers work with services that block virtual numbers — Google, Stripe, WhatsApp, and more.

## Installation

```bash
pip install agentsim-sdk
```

## Setup

Get an API key at [agentsim.dev/dashboard](https://agentsim.dev/dashboard):

```bash
export AGENTSIM_API_KEY=asm_live_...
```

## Tools

| Tool | Description |
|------|-------------|
| `AgentSIMProvisionTool` | Provision a real US mobile number, returns E.164 number + session ID |
| `AgentSIMWaitForOtpTool` | Wait for an OTP to arrive on a provisioned number |
| `AgentSIMReleaseTool` | Release a number back to the pool (always call after verification) |

## Usage

```python
from crewai import Agent, Crew, Task
from crewai_tools.tools.agentsim_tool import (
    AgentSIMProvisionTool,
    AgentSIMWaitForOtpTool,
    AgentSIMReleaseTool,
)

tools = [AgentSIMProvisionTool(), AgentSIMWaitForOtpTool(), AgentSIMReleaseTool()]

agent = Agent(
    role="Phone Verification Specialist",
    goal="Verify accounts using real carrier-grade phone numbers",
    tools=tools,
)

task = Task(
    description=(
        "Provision a US phone number, wait for OTP, then release the number. "
        "Report the phone number and OTP code received."
    ),
    expected_output="Phone number, OTP code, and release confirmation",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

## Pricing

- **Hobby**: 10 free sessions/month
- **Pay-as-you-go**: $0.99/session
- OTP-timeout sessions are not billed

## Links

- [AgentSIM Documentation](https://agentsim.dev)
- [Python SDK](https://github.com/agentsimdev/agentsim-python)
- [MCP Server](https://github.com/agentsimdev/agentsim-mcp)
- [More Examples](https://github.com/agentsimdev/agentsim-examples/tree/main/crewai-verification)
