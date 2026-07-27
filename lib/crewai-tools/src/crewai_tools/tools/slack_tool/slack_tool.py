import logging
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


def _require_slack_sdk() -> Any:
    try:
        from slack_sdk import WebClient

        return WebClient
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'slack-sdk'. Install with: \n"
            "  uv add crewai-tools --extra slack-sdk\n"
            "or\n"
            "  pip install slack-sdk\n"
        ) from exc


def _require_slack_bot_token() -> str:
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise ValueError(
            "Environment variable SLACK_BOT_TOKEN is required for Slack tools. "
            "Create a Slack app with a bot token (scopes: chat:write for sending "
            "messages, channels:history/groups:history for reading history) and "
            "set SLACK_BOT_TOKEN in your environment."
        )
    return token


class SlackSendMessageToolSchema(BaseModel):
    """Input for SlackSendMessageTool."""

    channel: str = Field(
        ...,
        description="Slack channel to post to, as a channel ID (e.g. 'C0123456789') "
        "or a name (e.g. '#general').",
    )
    message: str = Field(..., description="Message text to post to the channel.")
    thread_ts: str | None = Field(
        None,
        description="Timestamp of a parent message to reply in a thread. "
        "Omit to post a new top-level message.",
    )


class SlackSendMessageTool(BaseTool):
    name: str = "Send Slack Message"
    description: str = (
        "Posts a message to a Slack channel. Use this to notify a channel or "
        "reply in a thread. Requires a Slack bot token with the chat:write scope."
    )
    args_schema: type[BaseModel] = SlackSendMessageToolSchema

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SLACK_BOT_TOKEN",
                description="Slack bot token (xoxb-...) with chat:write scope",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["slack-sdk"])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _require_slack_sdk()
        _require_slack_bot_token()

    def _run(
        self, channel: str, message: str, thread_ts: str | None = None, **_: Any
    ) -> str:
        web_client_cls = _require_slack_sdk()
        client = web_client_cls(token=_require_slack_bot_token())

        try:
            from slack_sdk.errors import SlackApiError

            response = client.chat_postMessage(
                channel=channel, text=message, thread_ts=thread_ts
            )
            return (
                f"Message posted to {channel} "
                f"(ts={response.get('ts')}, thread_ts={thread_ts or response.get('ts')})."
            )
        except SlackApiError as e:
            error_code = e.response.get("error", "unknown_error")
            return f"Slack API error while posting to {channel}: {error_code}"
        except Exception as e:
            return f"Unexpected error posting Slack message to {channel}: {e}"

    async def _arun(
        self, channel: str, message: str, thread_ts: str | None = None, **kwargs: Any
    ) -> str:
        return self._run(
            channel=channel, message=message, thread_ts=thread_ts, **kwargs
        )


class SlackChannelHistoryToolSchema(BaseModel):
    """Input for SlackChannelHistoryTool."""

    channel: str = Field(
        ...,
        description="Slack channel ID to read history from (e.g. 'C0123456789'). "
        "Slack's conversations.history API requires a channel ID, not a name.",
    )
    limit: int = Field(
        20,
        ge=1,
        le=200,
        description="Maximum number of messages to fetch, most recent first.",
    )


class SlackChannelHistoryTool(BaseTool):
    name: str = "Read Slack Channel History"
    description: str = (
        "Fetches recent messages from a Slack channel. Use this to give an agent "
        "context on a conversation before it responds or takes action. Requires a "
        "Slack bot token with the channels:history (or groups:history for private "
        "channels) scope, and the bot must be a member of the channel."
    )
    args_schema: type[BaseModel] = SlackChannelHistoryToolSchema

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SLACK_BOT_TOKEN",
                description="Slack bot token (xoxb-...) with channels:history scope",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["slack-sdk"])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _require_slack_sdk()
        _require_slack_bot_token()

    def _run(self, channel: str, limit: int = 20, **_: Any) -> str:
        web_client_cls = _require_slack_sdk()
        client = web_client_cls(token=_require_slack_bot_token())

        try:
            from slack_sdk.errors import SlackApiError

            response = client.conversations_history(channel=channel, limit=limit)
            messages = response.get("messages", [])
            if not messages:
                return f"No messages found in channel {channel}."

            lines = []
            for msg in messages:
                user = msg.get("user", "unknown")
                text = msg.get("text", "")
                ts = msg.get("ts", "")
                lines.append(f"[{ts}] {user}: {text}")
            return "\n".join(lines)
        except SlackApiError as e:
            error_code = e.response.get("error", "unknown_error")
            return f"Slack API error while reading history for {channel}: {error_code}"
        except Exception as e:
            return f"Unexpected error reading Slack channel history for {channel}: {e}"

    async def _arun(self, channel: str, limit: int = 20, **kwargs: Any) -> str:
        return self._run(channel=channel, limit=limit, **kwargs)
