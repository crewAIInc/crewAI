"""CrewAI Platform application catalog."""

from typing import Final, Literal, get_args


PlatformApp = Literal[
    "gmail",
    "github",
    "google_drive",
    "google_sheets",
    "slack",
    "google_calendar",
    "whatsapp",
    "youtube",
    "instagram",
    "outlook",
    "google_docs",
    "linkedin",
    "asana",
    "box",
    "clickup",
    "hubspot",
    "jira",
    "linear",
    "notion",
    "salesforce",
    "shopify",
    "stripe",
    "zendesk",
]

PLATFORM_APPS: Final[tuple[str, ...]] = (*get_args(PlatformApp),)
