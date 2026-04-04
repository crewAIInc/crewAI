import os
from typing import Any

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


API_BASE = "https://api.multimail.dev"


class _MultiMailBase(BaseTool):
    """Base class for MultiMail tools."""

    api_key: str | None = None
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="MULTIMAIL_API_KEY",
                description="MultiMail API key",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("MULTIMAIL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either through constructor or MULTIMAIL_API_KEY environment variable"
            )

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        resp = requests.get(f"{API_BASE}{path}", headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict | None = None) -> dict[str, Any]:
        resp = requests.post(f"{API_BASE}{path}", headers=self._headers(), json=json)
        resp.raise_for_status()
        return resp.json()


# --- Input schemas ---

class CheckInboxInput(BaseModel):
    mailbox: str = Field(description="Mailbox address to check")
    status: str | None = Field(default=None, description="Filter by status (e.g. 'unread')")
    limit: int | None = Field(default=None, description="Max number of emails to return")


class ReadEmailInput(BaseModel):
    email_id: str = Field(description="ID of the email to read")


class SendEmailInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body (plain text or HTML)")
    mailbox: str = Field(description="Mailbox to send from")
    cc: str | None = Field(default=None, description="CC recipient(s)")
    bcc: str | None = Field(default=None, description="BCC recipient(s)")
    reply_to: str | None = Field(default=None, description="Reply-to address")


class ReplyEmailInput(BaseModel):
    email_id: str = Field(description="ID of the email to reply to")
    body: str = Field(description="Reply body")
    mailbox: str = Field(description="Mailbox to send from")


class SearchContactsInput(BaseModel):
    query: str | None = Field(default=None, description="Search query")
    mailbox: str | None = Field(default=None, description="Filter by mailbox")


class ListPendingInput(BaseModel):
    mailbox: str | None = Field(default=None, description="Filter by mailbox")


class DecideEmailInput(BaseModel):
    email_id: str = Field(description="ID of the pending email")
    decision: str = Field(description="'approve' or 'reject'")
    reason: str | None = Field(default=None, description="Reason for decision")


class GetThreadInput(BaseModel):
    thread_id: str = Field(description="Thread ID to retrieve")


class TagEmailInput(BaseModel):
    email_id: str = Field(description="ID of the email to tag")
    tags: list[str] = Field(description="List of tags to add")


# --- Tools ---

class MultiMailCheckInboxTool(_MultiMailBase):
    name: str = "MultiMail Check Inbox"
    description: str = "Check a MultiMail mailbox inbox. Returns a list of emails with optional status filtering."
    args_schema: type[BaseModel] = CheckInboxInput

    def _run(self, mailbox: str, status: str | None = None, limit: int | None = None) -> str:
        params = {"mailbox": mailbox}
        if status:
            params["status"] = status
        if limit:
            params["limit"] = str(limit)
        return str(self._get("/v1/emails/inbox", params))


class MultiMailReadEmailTool(_MultiMailBase):
    name: str = "MultiMail Read Email"
    description: str = "Read a specific email by ID. Returns the full email content including headers, body, and metadata."
    args_schema: type[BaseModel] = ReadEmailInput

    def _run(self, email_id: str) -> str:
        return str(self._get(f"/v1/emails/{email_id}"))


class MultiMailSendEmailTool(_MultiMailBase):
    name: str = "MultiMail Send Email"
    description: str = "Send a new email through MultiMail. Subject to the mailbox's oversight mode."
    args_schema: type[BaseModel] = SendEmailInput

    def _run(self, to: str, subject: str, body: str, mailbox: str, cc: str | None = None, bcc: str | None = None, reply_to: str | None = None) -> str:
        payload = {"to": to, "subject": subject, "body": body, "mailbox": mailbox}
        if cc:
            payload["cc"] = cc
        if bcc:
            payload["bcc"] = bcc
        if reply_to:
            payload["reply_to"] = reply_to
        return str(self._post("/v1/emails/send", payload))


class MultiMailReplyEmailTool(_MultiMailBase):
    name: str = "MultiMail Reply Email"
    description: str = "Reply to an existing email. Subject to the mailbox's oversight mode."
    args_schema: type[BaseModel] = ReplyEmailInput

    def _run(self, email_id: str, body: str, mailbox: str) -> str:
        return str(self._post(f"/v1/emails/{email_id}/reply", {"body": body, "mailbox": mailbox}))


class MultiMailSearchContactsTool(_MultiMailBase):
    name: str = "MultiMail Search Contacts"
    description: str = "Search the contact list with an optional query and mailbox filter."
    args_schema: type[BaseModel] = SearchContactsInput

    def _run(self, query: str | None = None, mailbox: str | None = None) -> str:
        params = {}
        if query:
            params["query"] = query
        if mailbox:
            params["mailbox"] = mailbox
        return str(self._get("/v1/contacts", params))


class MultiMailListPendingTool(_MultiMailBase):
    name: str = "MultiMail List Pending"
    description: str = "List emails pending human approval. Use with gated oversight modes."
    args_schema: type[BaseModel] = ListPendingInput

    def _run(self, mailbox: str | None = None) -> str:
        params = {}
        if mailbox:
            params["mailbox"] = mailbox
        return str(self._get("/v1/emails/pending", params))


class MultiMailDecideEmailTool(_MultiMailBase):
    name: str = "MultiMail Decide Email"
    description: str = "Approve or reject a pending email. Only applies to emails in gated oversight modes."
    args_schema: type[BaseModel] = DecideEmailInput

    def _run(self, email_id: str, decision: str, reason: str | None = None) -> str:
        payload = {"decision": decision}
        if reason:
            payload["reason"] = reason
        return str(self._post(f"/v1/emails/{email_id}/decide", payload))


class MultiMailGetThreadTool(_MultiMailBase):
    name: str = "MultiMail Get Thread"
    description: str = "Retrieve a full email thread by thread ID."
    args_schema: type[BaseModel] = GetThreadInput

    def _run(self, thread_id: str) -> str:
        return str(self._get(f"/v1/emails/thread/{thread_id}"))


class MultiMailTagEmailTool(_MultiMailBase):
    name: str = "MultiMail Tag Email"
    description: str = "Add tags to an email for organization and filtering."
    args_schema: type[BaseModel] = TagEmailInput

    def _run(self, email_id: str, tags: list[str]) -> str:
        return str(self._post(f"/v1/emails/{email_id}/tag", {"tags": tags}))
