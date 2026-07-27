# Slack Tools Documentation

## Description
`SlackSendMessageTool` and `SlackChannelHistoryTool` let CrewAI agents interact with Slack: posting notifications or replies, and reading recent channel messages for context. Both use Slack's Web API via `slack_sdk`.

## Features
- **SlackSendMessageTool**: Post a message to a channel, optionally as a threaded reply (`thread_ts`)
- **SlackChannelHistoryTool**: Fetch recent messages from a channel, most recent first

## Installation
```shell
pip install 'crewai[tools]'
uv add crewai-tools --extra slack-sdk
```

## Usage
```python
from crewai_tools import SlackSendMessageTool, SlackChannelHistoryTool

send_tool = SlackSendMessageTool()
send_tool._run(channel="#general", message="Deployment finished successfully.")

history_tool = SlackChannelHistoryTool()
history_tool._run(channel="C0123456789", limit=20)
```

## Configuration
1. **Create a Slack app** at https://api.slack.com/apps and install it to your workspace.
2. **Grant bot token scopes**:
   - `chat:write` — required for `SlackSendMessageTool`
   - `channels:history` (public channels) and/or `groups:history` (private channels) — required for `SlackChannelHistoryTool`
3. **Invite the bot** to any channel it needs to post to or read from.
4. Set the environment variable `SLACK_BOT_TOKEN` to the bot's `xoxb-...` token.

## Notes
- `SlackChannelHistoryTool` requires a channel ID (e.g. `C0123456789`), since Slack's `conversations.history` API does not accept channel names.
- `SlackSendMessageTool` accepts either a channel ID or a name (e.g. `#general`).
- Both tools return a descriptive string on Slack API errors rather than raising, so agents can react to failures (e.g. `channel_not_found`, `not_in_channel`) instead of crashing.
