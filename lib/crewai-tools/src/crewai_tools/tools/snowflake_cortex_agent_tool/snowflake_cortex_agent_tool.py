from __future__ import annotations

import json
import logging
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr
import requests


logger = logging.getLogger(__name__)


class SnowflakeCortexAgentToolInput(BaseModel):
    """Input schema for SnowflakeCortexAgentTool."""

    model_config = ConfigDict(protected_namespaces=())

    query: str = Field(
        ...,
        description=(
            "The natural language data question to ask the Cortex Agent. "
            "The agent will plan, route to Cortex Analyst (text-to-SQL on "
            "structured data) or Cortex Search (retrieval over unstructured "
            "data), execute, and return a final answer."
        ),
    )


class SnowflakeCortexAgentTool(BaseTool):
    """Tool for delegating data questions to a Snowflake Cortex Agent.

    Snowflake Cortex Agents orchestrate across structured (Cortex Analyst) and
    unstructured (Cortex Search) data sources inside Snowflake's secure
    perimeter. Instead of having a CrewAI agent generate SQL or pick between
    retrieval and analytics, this tool sends the natural language question to a
    Cortex Agent and returns its final answer for use in downstream steps.

    See: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents

    There are two ways to configure the tool:

    1. Reference an existing **agent object** by passing ``database``,
       ``snowflake_schema`` and ``agent_name``. The tool calls
       ``POST /api/v2/databases/{database}/schemas/{schema}/agents/{name}:run``.
    2. Run **without an agent object** by passing ``tools`` (and optionally
       ``tool_resources``, ``models``, ``instructions`` and ``orchestration``).
       The tool calls ``POST /api/v2/cortex/agent:run``.

    Authentication uses a bearer token (Snowflake Programmatic Access Token,
    JWT, or OAuth token). The token may be passed via ``auth_token`` or via
    the ``SNOWFLAKE_CORTEX_AGENT_TOKEN`` environment variable. The Snowflake
    account identifier may be passed via ``account`` or via the
    ``SNOWFLAKE_ACCOUNT`` environment variable. To target a custom hostname
    (for example, a private link endpoint), set ``host`` directly.

    Example::

        from crewai_tools import SnowflakeCortexAgentTool

        tool = SnowflakeCortexAgentTool(
            account="myorg-myaccount",
            auth_token="<programmatic-access-token>",
            database="MY_DB",
            snowflake_schema="MY_SCHEMA",
            agent_name="SALES_AGENT",
        )

        answer = tool.run(query="What was total revenue last quarter?")
    """

    name: str = "Snowflake Cortex Agent"
    description: str = (
        "Delegate a natural language data question to a Snowflake Cortex "
        "Agent. The agent reasons over structured data via Cortex Analyst "
        "(text-to-SQL with semantic models) and unstructured data via Cortex "
        "Search, then returns a final answer. Use this whenever a question "
        "is best answered with governed, retrieval-augmented analysis of "
        "Snowflake data instead of writing raw SQL."
    )
    args_schema: type[BaseModel] = SnowflakeCortexAgentToolInput

    # Connection configuration
    account: str | None = Field(
        default=None,
        description=(
            "Snowflake account identifier (e.g. 'myorg-myaccount'). Used to "
            "build the API hostname. Falls back to the SNOWFLAKE_ACCOUNT "
            "environment variable. Ignored when 'host' is set."
        ),
    )
    host: str | None = Field(
        default=None,
        description=(
            "Override the API hostname (e.g. 'myorg-myaccount.snowflake"
            "computing.com' or a private link host). When provided, takes "
            "precedence over 'account'."
        ),
    )
    auth_token: SecretStr | None = Field(
        default=None,
        description=(
            "Bearer token used for authentication (Programmatic Access "
            "Token, OAuth token, or JWT). Falls back to the "
            "SNOWFLAKE_CORTEX_AGENT_TOKEN environment variable."
        ),
    )

    # Agent-object configuration (optional; if all three are set, the agent
    # object endpoint is used).
    database: str | None = Field(
        default=None,
        description="Database containing the agent object.",
    )
    snowflake_schema: str | None = Field(
        default=None,
        description="Schema containing the agent object.",
    )
    agent_name: str | None = Field(
        default=None,
        description="Name of the agent object to invoke.",
    )

    # Inline configuration (used when an agent object is not referenced).
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "List of tool specifications when calling the agent without an "
            "agent object. See the Cortex Agents Run API docs for the schema."
        ),
    )
    tool_resources: dict[str, Any] | None = Field(
        default=None,
        description="Per-tool resource configuration keyed by tool name.",
    )
    tool_choice: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Tool selection policy (e.g. {'type': 'auto'} or "
            "{'type': 'required', 'name': ['analyst_tool']})."
        ),
    )
    models: dict[str, Any] | None = Field(
        default=None,
        description="Model configuration (e.g. {'orchestration': 'claude-4-sonnet'}).",
    )
    instructions: dict[str, Any] | None = Field(
        default=None,
        description="Agent instructions (response, orchestration, system, sample_questions).",
    )
    orchestration: dict[str, Any] | None = Field(
        default=None,
        description="Orchestration configuration such as budget constraints.",
    )

    # Request behaviour
    timeout: int = Field(
        default=600,
        description="Per-request timeout in seconds (Cortex Agent server timeout is 15 minutes).",
    )

    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SNOWFLAKE_ACCOUNT",
                description="Snowflake account identifier used to build the Cortex Agent API hostname.",
                required=False,
            ),
            EnvVar(
                name="SNOWFLAKE_CORTEX_AGENT_TOKEN",
                description="Bearer token (PAT, OAuth, or JWT) for the Cortex Agent REST API.",
                required=False,
            ),
        ]
    )

    _session: requests.Session | None = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._session = requests.Session()
        self._validate_credentials()

    def _validate_credentials(self) -> None:
        """Validate that we have enough information to make a request."""
        if not self._resolve_token():
            raise ValueError(
                "Snowflake Cortex Agent requires a bearer token. Pass "
                "'auth_token' or set the SNOWFLAKE_CORTEX_AGENT_TOKEN "
                "environment variable."
            )
        if not self._resolve_host():
            raise ValueError(
                "Snowflake Cortex Agent requires either 'host' or 'account' "
                "(or the SNOWFLAKE_ACCOUNT environment variable) to build "
                "the API URL."
            )
        agent_object_fields = (self.database, self.snowflake_schema, self.agent_name)
        any_set = any(agent_object_fields)
        all_set = all(agent_object_fields)
        if any_set and not all_set:
            raise ValueError(
                "To reference an agent object, all of 'database', "
                "'snowflake_schema' and 'agent_name' must be provided."
            )
        if not all_set and not self.tools:
            raise ValueError(
                "Provide either ('database', 'snowflake_schema', "
                "'agent_name') to reference an existing Cortex Agent object, "
                "or 'tools' (list of tool specs) to run an agent inline."
            )

    def _resolve_token(self) -> str | None:
        if self.auth_token is not None:
            value = self.auth_token.get_secret_value()
            if value:
                return value
        return os.environ.get("SNOWFLAKE_CORTEX_AGENT_TOKEN") or None

    def _resolve_host(self) -> str | None:
        if self.host:
            return self.host.strip().rstrip("/")
        account = self.account or os.environ.get("SNOWFLAKE_ACCOUNT")
        if not account:
            return None
        account = account.strip()
        return f"{account}.snowflakecomputing.com"

    def _build_url(self) -> str:
        host = self._resolve_host()
        if not host:
            raise ValueError("Snowflake host is not configured")
        scheme = "https://"
        if host.startswith(("http://", "https://")):
            scheme = ""
        if self.database and self.snowflake_schema and self.agent_name:
            return (
                f"{scheme}{host}/api/v2/databases/{self.database}/schemas/"
                f"{self.snowflake_schema}/agents/{self.agent_name}:run"
            )
        return f"{scheme}{host}/api/v2/cortex/agent:run"

    def _build_payload(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                }
            ],
            "stream": False,
        }
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice
        if self.models is not None:
            payload["models"] = self.models
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.orchestration is not None:
            payload["orchestration"] = self.orchestration
        if not (self.database and self.snowflake_schema and self.agent_name):
            if self.tools is not None:
                payload["tools"] = self.tools
            if self.tool_resources is not None:
                payload["tool_resources"] = self.tool_resources
        return payload

    def _build_headers(self) -> dict[str, str]:
        token = self._resolve_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        """Best-effort extraction of the assistant's textual answer."""
        content = response_json.get("content")
        if isinstance(content, list):
            texts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)
            if texts:
                return "\n".join(texts)
        # Fallback: serialize the entire response so the caller still gets
        # something useful (tool calls, citations, etc.).
        return json.dumps(response_json, ensure_ascii=False)

    def _run(self, query: str, **_kwargs: Any) -> str:
        url = self._build_url()
        payload = self._build_payload(query)
        headers = self._build_headers()
        session = self._session or requests.Session()
        try:
            response = session.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as e:
            logger.error("Cortex Agent request failed: %s", e)
            return f"Error calling Snowflake Cortex Agent: {e}"
        if response.status_code >= 400:
            logger.error(
                "Cortex Agent returned %s: %s", response.status_code, response.text
            )
            return (
                f"Snowflake Cortex Agent returned HTTP {response.status_code}: "
                f"{response.text}"
            )
        try:
            data = response.json()
        except ValueError:
            return response.text
        if not isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False)
        return self._extract_text(data)
