# Spix Tools for CrewAI

Give your CrewAI agents a real phone number and voice. Make outbound phone calls,
send SMS, and send email — all from a simple CrewAI tool interface.

[Spix](https://spix.sh) is communications infrastructure for AI agents: real phone
numbers, AI voice calls (~500ms latency using Deepgram Nova-3 + Claude + Cartesia
Sonic-3), SMS, and email.

## Installation

```bash
pip install crewai-tools httpx
```

Get a Spix API key at [app.spix.sh/api-keys](https://app.spix.sh/api-keys).

## Available Tools

| Tool | Description |
|------|-------------|
| `SpixCallTool` | Place an outbound AI phone call using a playbook |
| `SpixSMSTool` | Send an SMS message |
| `SpixEmailTool` | Send an email |

## Quick Start

```python
import os
from crewai import Agent, Crew, Task
from crewai_tools import SpixCallTool, SpixSMSTool, SpixEmailTool

os.environ["SPIX_API_KEY"] = "your-api-key"

call_tool = SpixCallTool()
sms_tool = SpixSMSTool()
email_tool = SpixEmailTool()

outreach_agent = Agent(
    role="Outreach Specialist",
    goal="Contact leads via the most appropriate channel",
    backstory="You are an expert at multi-channel outreach.",
    tools=[call_tool, sms_tool, email_tool],
    verbose=True,
)

task = Task(
    description=(
        "Call +19175550123 using playbook cmp_call_abc123 from +14155550101 "
        "to confirm their demo appointment."
    ),
    expected_output="Confirmation that the call was placed with the session ID.",
    agent=outreach_agent,
)

crew = Crew(agents=[outreach_agent], tasks=[task])
result = crew.kickoff()
```

## Setup: Phone Numbers & Playbooks

1. **Rent a phone number** at [app.spix.sh](https://app.spix.sh)
2. **Create a playbook** (defines your AI persona, voice, and script)
3. **Bind the number** to the playbook

## API Key

Pass the key directly or set the `SPIX_API_KEY` environment variable:

```python
# Via env var (recommended)
os.environ["SPIX_API_KEY"] = "sk_..."

# Or pass directly
tool = SpixCallTool(api_key="sk_...")
```

## Links

- [Spix Dashboard](https://app.spix.sh)
- [API Docs](https://spix.sh/docs)
- [MCP Server](https://github.com/Spix-HQ/spix-mcp) — use Spix from Claude Desktop and any MCP-compatible host
