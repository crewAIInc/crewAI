from __future__ import annotations

import os
import time
from typing import Any, cast

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crewai_tools.oci.common import create_oci_client_kwargs, get_oci_module


class OCIGenAIInvokeAgentToolInput(BaseModel):
    """Input schema for OCIGenAIInvokeAgentTool."""

    query: str = Field(..., description="The query to send to the OCI Generative AI agent")


class OCIGenAIInvokeAgentTool(BaseTool):
    name: str = "OCI Generative AI Agent Invoke Tool"
    description: str = (
        "Invokes an Oracle Cloud Infrastructure Generative AI agent endpoint."
    )
    args_schema: type[BaseModel] = OCIGenAIInvokeAgentToolInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["oci"])
    agent_endpoint_id: str | None = None
    session_id: str | None = None
    create_session_if_missing: bool = True
    client: Any | None = None

    def __init__(
        self,
        agent_endpoint_id: str | None = None,
        session_id: str | None = None,
        create_session_if_missing: bool = True,
        description: str | None = None,
        *,
        auth_type: str = "API_KEY",
        auth_profile: str | None = None,
        auth_file_location: str | None = None,
        service_endpoint: str | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent_endpoint_id = agent_endpoint_id or os.getenv(
            "OCI_AGENT_ENDPOINT_ID"
        )
        self.session_id = session_id
        self.create_session_if_missing = create_session_if_missing
        self.client = client

        if description:
            self.description = description

        if not self.agent_endpoint_id:
            raise ValueError(
                "agent_endpoint_id is required. Set it explicitly or use OCI_AGENT_ENDPOINT_ID."
            )

        if self.client is None:
            oci = get_oci_module()
            resolved_auth_profile = cast(
                str, auth_profile or os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
            )
            resolved_auth_file_location = cast(
                str,
                auth_file_location or os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config"),
            )
            client_kwargs = create_oci_client_kwargs(
                auth_type=auth_type,
                auth_profile=resolved_auth_profile,
                auth_file_location=resolved_auth_file_location,
                service_endpoint=service_endpoint
                or os.getenv("OCI_AGENT_RUNTIME_ENDPOINT"),
                timeout=(10, 180),
            )
            self.client = oci.generative_ai_agent_runtime.GenerativeAiAgentRuntimeClient(
                **client_kwargs
            )

    def _require_client(self) -> Any:
        if self.client is None:
            raise ValueError("OCI Generative AI agent client is not initialized.")
        return self.client

    def _ensure_session_id(self) -> str | None:
        if self.session_id or not self.create_session_if_missing:
            return self.session_id

        oci = get_oci_module()
        response = self._require_client().create_session(
            agent_endpoint_id=self.agent_endpoint_id,
            create_session_details=oci.generative_ai_agent_runtime.models.CreateSessionDetails(
                display_name=f"crewai-tools-{int(time.time())}",
                description="Created by CrewAI OCI agent tool",
            ),
        )
        self.session_id = str(response.data.id)
        return self.session_id

    def _extract_text(self, response: Any) -> str:
        message = getattr(response.data, "message", None)
        content = getattr(message, "content", None)
        if content is not None and getattr(content, "text", None):
            return str(content.text)

        required_actions = getattr(response.data, "required_actions", None) or []
        if required_actions:
            return (
                "OCI agent requires follow-up actions before it can complete the response."
            )

        return str(response.data)

    def _run(self, query: str) -> str:
        try:
            oci = get_oci_module()
            session_id = self._ensure_session_id()
            chat_details = oci.generative_ai_agent_runtime.models.ChatDetails(
                user_message=query,
                session_id=session_id,
                should_stream=False,
            )
            response = self._require_client().chat(
                agent_endpoint_id=self.agent_endpoint_id,
                chat_details=chat_details,
            )
            return self._extract_text(response)
        except Exception as error:
            return f"Error invoking OCI Generative AI agent: {error!s}"
