"""CrewAI Platform application catalog."""

from typing import Final, Literal, get_args


PlatformApp = Literal[
    "asana",
    "box",
    "clickup",
    "github",
    "gmail",
    "google_calendar",
    "google_sheets",
    "hubspot",
    "jira",
    "linear",
    "notion",
    "salesforce",
    "shopify",
    "slack",
    "stripe",
    "zendesk",
]

PLATFORM_APPS: Final[tuple[str, ...]] = (*get_args(PlatformApp),)
