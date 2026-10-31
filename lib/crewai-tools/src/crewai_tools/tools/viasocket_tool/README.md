# ViaSocketTools

## Description

[viaSocket](https://viasocket.com) connects 2,300+ apps (Slack, Google Sheets, HubSpot, Notion and more). `ViaSocketTools` turns each chosen action of an app your end user connected into a CrewAI tool, run on that user's own connection, so your agents act in their apps without connector code or OAuth on your side.

## Installation

```shell
uv add viasocket
```

## Environment Variables

- `VIASOCKET_ORG_ID`, `VIASOCKET_PROJECT_ID`, `VIASOCKET_EMBED_SECRET` (required): run `npx viasocket-apps setup` in your project, or copy them from the viaSocket dashboard (**Integrations → Create embed → Install Code**).

To try it before your product has a connect button, connect an app for yourself: `npx viasocket-apps connect slack --user me`.

## Example

```python
from crewai import Agent
from crewai_tools import ViaSocketTools

tools = ViaSocketTools(user="me", apps=["slack"], actions={"slack": ["send-message"]})

agent = Agent(
    role="Operations assistant",
    goal="Keep the team updated in Slack",
    backstory="You post updates where the team reads them.",
    tools=tools,
)
```

Fields that hold an id (a channel, a sheet) are filled through the app's `<app>_list-options` tool, which lists the real values from the user's account. More: [viaSocket Python package](https://github.com/Walkover-Web-Solution/viasocket-apps/tree/main/python#readme).
