"""CrewAI AMP platform tools exposed by the crew-creation wizard."""

from crewai_core.platform_apps import PLATFORM_APPS


PLATFORM_TOOL_PREFIX = "platform:"

_APP_DESCRIPTIONS: dict[str, str] = {
    "gmail": "Gmail Integration",
    "github": "GitHub Integration",
    "google_drive": "Google Drive Integration",
    "google_sheets": "Google Sheets Integration",
    "slack": "Slack Integration",
    "google_calendar": "Google Calendar Integration",
    "whatsapp": "WhatsApp Integration",
    "youtube": "YouTube Integration",
    "instagram": "Instagram Integration",
    "outlook": "Outlook Integration",
    "google_docs": "Google Docs Integration",
    "linkedin": "LinkedIn Integration",
    "asana": "Asana Integration",
    "box": "Box Integration",
    "clickup": "ClickUp Integration",
    "hubspot": "HubSpot Integration",
    "jira": "Jira Integration",
    "linear": "Linear Integration",
    "notion": "Notion Integration",
    "salesforce": "Salesforce Integration",
    "shopify": "Shopify Integration",
    "stripe": "Stripe Integration",
    "zendesk": "Zendesk Integration",
}

PLATFORM_TOOLS: list[tuple[str, str]] = [
    (f"{PLATFORM_TOOL_PREFIX}{app}", _APP_DESCRIPTIONS[app]) for app in PLATFORM_APPS
]
