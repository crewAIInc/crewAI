# MultiMail Tool

The MultiMail tools give CrewAI agents email capabilities through [MultiMail](https://multimail.dev) — an email platform built for AI agents with graduated oversight modes.

## Installation

```bash
pip install crewai-tools requests
```

## Configuration

Set your MultiMail API key:

```bash
export MULTIMAIL_API_KEY="mm_live_..."
```

Or pass it directly:

```python
tool = MultiMailCheckInboxTool(api_key="mm_live_...")
```

## Available Tools

| Tool | Description |
|------|-------------|
| `MultiMailCheckInboxTool` | Check a mailbox inbox |
| `MultiMailReadEmailTool` | Read a specific email |
| `MultiMailSendEmailTool` | Send a new email |
| `MultiMailReplyEmailTool` | Reply to an email |
| `MultiMailSearchContactsTool` | Search contacts |
| `MultiMailListPendingTool` | List pending approvals |
| `MultiMailDecideEmailTool` | Approve/reject pending email |
| `MultiMailGetThreadTool` | Get a full thread |
| `MultiMailTagEmailTool` | Tag an email |

## Usage

```python
from crewai import Agent, Task, Crew
from crewai_tools import MultiMailCheckInboxTool, MultiMailReadEmailTool, MultiMailSendEmailTool

check_inbox = MultiMailCheckInboxTool()
read_email = MultiMailReadEmailTool()
send_email = MultiMailSendEmailTool()

email_agent = Agent(
    role="Email Assistant",
    goal="Help manage email communications",
    backstory="You are a helpful assistant that manages email.",
    tools=[check_inbox, read_email, send_email],
)

task = Task(
    description="Check the inbox for support@example.com and summarize unread emails",
    expected_output="A summary of unread emails",
    agent=email_agent,
)

crew = Crew(agents=[email_agent], tasks=[task])
result = crew.kickoff()
```

## Oversight Modes

MultiMail supports graduated trust levels:

- **gated_all** — All actions require human approval
- **gated_send** — Sends require approval; reads are automatic
- **monitored** — Agent acts autonomously; human can review
- **autonomous** — Full autonomous operation

## Links

- [MultiMail](https://multimail.dev)
- [PyPI](https://pypi.org/project/crewai-multimail/)
- [GitHub](https://github.com/multimail-dev/crewai-multimail)
