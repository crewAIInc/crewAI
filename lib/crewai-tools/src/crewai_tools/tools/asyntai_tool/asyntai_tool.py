import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class AsyntaiBaseTool(BaseTool):
    """Base class for the Asyntai tools.

    Asyntai (asyntai.com) is an AI chatbot for websites. It answers visitors from
    the website's own pages, documents and knowledge base, and it captures leads.
    The tools return a readable error text to the agent and do not raise.

    Attributes:
        api_key: Asyntai API key. Falls back to the ASYNTAI_API_KEY env variable.
        website_id: Default Asyntai website. Without it, the primary website is used.
        api_base_url: Base URL of the Asyntai API.
        timeout: Timeout in seconds for each API request.
    """

    api_key: str | None = Field(default=None, repr=False)
    website_id: int | None = None
    api_base_url: str = "https://asyntai.com/api/v1"
    timeout: int = 60
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="ASYNTAI_API_KEY",
                description="API key for Asyntai",
                required=True,
            ),
        ]
    )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call the Asyntai API and return the JSON body.

        Raises:
            RuntimeError: With a readable message when the API refuses the call.
        """
        api_key = self.api_key or os.getenv("ASYNTAI_API_KEY")
        if not api_key:
            raise RuntimeError("ASYNTAI_API_KEY not set")

        body = (
            {k: v for k, v in payload.items() if v is not None}
            if payload is not None
            else None
        )
        query = (
            {k: v for k, v in params.items() if v is not None}
            if params is not None
            else None
        )
        response = requests.request(
            method,
            f"{self.api_base_url.rstrip('/')}{path}",
            json=body,
            params=query,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "crewai-asyntai-tools",
            },
            timeout=self.timeout,
        )
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                f"Asyntai returned a non-JSON response (HTTP {response.status_code})"
            ) from None
        if not isinstance(data, dict):
            raise RuntimeError("Asyntai returned an unexpected response body")

        if response.status_code == 401:
            raise RuntimeError(
                "Asyntai rejected the API key. "
                "Check the key in the Asyntai dashboard under Settings > API"
            )
        if response.status_code == 403:
            raise RuntimeError(
                "The Asyntai API needs a paid plan (Starter or higher). "
                "See https://asyntai.com/pricing/"
            )
        if response.status_code == 429:
            raise RuntimeError("The Asyntai message limit for this month is reached")
        if not data.get("success", False):
            raise RuntimeError(
                data.get("error") or f"Asyntai API error (HTTP {response.status_code})"
            )
        return data

    def _website(self, website_id: int | None) -> int | None:
        return website_id if website_id is not None else self.website_id


class AsyntaiAskWebsiteSchema(BaseModel):
    question: str = Field(..., description="The question to ask, in natural language.")
    session_id: str | None = Field(
        default=None,
        description="Reuse the session id from an earlier answer to ask a follow-up question.",
    )
    website_id: int | None = Field(
        default=None,
        description="The Asyntai website to ask. Defaults to the configured or primary website.",
    )


class AsyntaiAskWebsiteTool(AsyntaiBaseTool):
    name: str = "Ask Asyntai Website"
    description: str = (
        "Asks a question to the Asyntai knowledge base of a website. The answer comes "
        "from the website's own pages, documents and knowledge base. Returns JSON with "
        "the answer and a session id for follow-up questions."
    )
    args_schema: type[BaseModel] = AsyntaiAskWebsiteSchema

    def _run(
        self,
        question: str,
        session_id: str | None = None,
        website_id: int | None = None,
    ) -> str:
        try:
            result = self._request(
                "POST",
                "/chat/",
                payload={
                    "message": question,
                    "session_id": session_id,
                    "website_id": self._website(website_id),
                },
            )
            return json.dumps(
                {
                    "answer": result.get("response", ""),
                    "session_id": result.get("session_id"),
                }
            )
        except Exception as e:
            return f"Error asking Asyntai: {e}"


class AsyntaiListWebsitesTool(AsyntaiBaseTool):
    name: str = "List Asyntai Websites"
    description: str = (
        "Lists the Asyntai websites (chatbots) on the account, with their ids."
    )

    def _run(self) -> str:
        try:
            return json.dumps(self._request("GET", "/websites/").get("websites", []))
        except Exception as e:
            return f"Error listing Asyntai websites: {e}"


class AsyntaiAddTextSchema(BaseModel):
    title: str = Field(..., description="A short title for the text.")
    content: str = Field(..., description="The text to add.")
    website_id: int | None = Field(
        default=None,
        description="The Asyntai website to add it to. Defaults to the configured or primary website.",
    )


class AsyntaiAddTextTool(AsyntaiBaseTool):
    name: str = "Add Text to Asyntai Knowledge"
    description: str = (
        "Adds a text to the Asyntai knowledge base, so the chatbot can answer from it."
    )
    args_schema: type[BaseModel] = AsyntaiAddTextSchema

    def _run(self, title: str, content: str, website_id: int | None = None) -> str:
        try:
            return json.dumps(
                self._request(
                    "POST",
                    "/knowledge/text/",
                    payload={
                        "title": title,
                        "content": content,
                        "website_id": self._website(website_id),
                    },
                )
            )
        except Exception as e:
            return f"Error adding text to Asyntai: {e}"


class AsyntaiAddUrlSchema(BaseModel):
    url: str = Field(..., description="The address of the page to add.")
    website_id: int | None = Field(
        default=None,
        description="The Asyntai website to add it to. Defaults to the configured or primary website.",
    )


class AsyntaiAddUrlTool(AsyntaiBaseTool):
    name: str = "Add URL to Asyntai Knowledge"
    description: str = (
        "Adds a web page to the Asyntai knowledge base. "
        "Asyntai reads the page and stores its content."
    )
    args_schema: type[BaseModel] = AsyntaiAddUrlSchema

    def _run(self, url: str, website_id: int | None = None) -> str:
        try:
            return json.dumps(
                self._request(
                    "POST",
                    "/knowledge/url/",
                    payload={"url": url, "website_id": self._website(website_id)},
                )
            )
        except Exception as e:
            return f"Error adding {url} to Asyntai: {e}"


class AsyntaiListKnowledgeSchema(BaseModel):
    limit: int = Field(default=50, description="Maximum number of entries to return.")
    website_id: int | None = Field(
        default=None,
        description="The Asyntai website to read. Defaults to the configured or primary website.",
    )


class AsyntaiListKnowledgeTool(AsyntaiBaseTool):
    name: str = "List Asyntai Knowledge"
    description: str = "Lists the entries in the Asyntai knowledge base."
    args_schema: type[BaseModel] = AsyntaiListKnowledgeSchema

    def _run(self, limit: int = 50, website_id: int | None = None) -> str:
        try:
            result = self._request(
                "GET",
                "/knowledge/",
                params={"limit": limit, "website_id": self._website(website_id)},
            )
            return json.dumps(result.get("entries", []))
        except Exception as e:
            return f"Error listing Asyntai knowledge: {e}"


class AsyntaiGetLeadsSchema(BaseModel):
    limit: int = Field(
        default=50, description="Maximum number of leads to return (at most 100)."
    )
    email: str | None = Field(
        default=None, description="Only return the lead with this email address."
    )
    since: str | None = Field(
        default=None,
        description="Only return leads captured after this date or time (ISO 8601, for example 2026-01-31).",
    )
    website_id: int | None = Field(
        default=None,
        description="The Asyntai website to read. Defaults to the configured website, or all websites.",
    )


class AsyntaiGetLeadsTool(AsyntaiBaseTool):
    name: str = "Get Asyntai Leads"
    description: str = (
        "Reads the leads that the Asyntai chatbot captured "
        "(visitors who left contact details)."
    )
    args_schema: type[BaseModel] = AsyntaiGetLeadsSchema

    def _run(
        self,
        limit: int = 50,
        email: str | None = None,
        since: str | None = None,
        website_id: int | None = None,
    ) -> str:
        try:
            result = self._request(
                "GET",
                "/leads/",
                params={
                    "limit": limit,
                    "email": email,
                    "since": since,
                    "website_id": self._website(website_id),
                },
            )
            return json.dumps(result.get("leads", []))
        except Exception as e:
            return f"Error reading Asyntai leads: {e}"


class AsyntaiGetConversationSchema(BaseModel):
    session_id: str = Field(
        ..., description="The id of the conversation, for example from a lead."
    )
    limit: int = Field(default=50, description="Maximum number of messages to return.")


class AsyntaiGetConversationTool(AsyntaiBaseTool):
    name: str = "Get Asyntai Conversation"
    description: str = "Reads the messages of one Asyntai chat conversation."
    args_schema: type[BaseModel] = AsyntaiGetConversationSchema

    def _run(self, session_id: str, limit: int = 50) -> str:
        try:
            result = self._request(
                "GET",
                "/conversations/",
                params={"session_id": session_id, "limit": limit},
            )
            return json.dumps(result.get("messages", []))
        except Exception as e:
            return f"Error reading Asyntai conversation {session_id}: {e}"
