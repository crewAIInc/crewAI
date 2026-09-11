"""CrewAI AMP platform tools exposed by the crew-creation wizard."""

from crewai_core.platform_apps import PLATFORM_APPS


PLATFORM_TOOL_PREFIX = "platform:"

_APP_DESCRIPTIONS: dict[str, str] = {
    "asana": "Asana Integration",
    "box": "Box Integration",
    "clickup": "ClickUp Integration",
    "github": "GitHub Integration",
    "gmail": "Gmail Integration",
    "google_calendar": "Google Calendar Integration",
    "google_sheets": "Google Sheets Integration",
    "hubspot": "HubSpot Integration",
    "jira": "Jira Integration",
    "linear": "Linear Integration",
    "notion": "Notion Integration",
    "salesforce": "Salesforce Integration",
    "shopify": "Shopify Integration",
    "slack": "Slack Integration",
    "stripe": "Stripe Integration",
    "zendesk": "Zendesk Integration",
}

PLATFORM_TOOLS: list[tuple[str, str]] = [
    (f"{PLATFORM_TOOL_PREFIX}{app}", _APP_DESCRIPTIONS[app]) for app in PLATFORM_APPS
]
