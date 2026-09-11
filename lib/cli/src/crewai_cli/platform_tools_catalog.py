"""CrewAI AMP platform tools exposed by the crew-creation wizard."""

from crewai_core.platform_apps import PLATFORM_APPS


PLATFORM_TOOL_PREFIX = "platform:"

_APP_DESCRIPTIONS: dict[str, str] = {
    "asana": "Work with Asana projects and tasks",
    "box": "Access and manage files in Box",
    "clickup": "Work with ClickUp tasks and workspaces",
    "github": "Work with GitHub repositories and issues",
    "gmail": "Read and send email with Gmail",
    "google_calendar": "Manage Google Calendar events",
    "google_sheets": "Read and update Google Sheets",
    "hubspot": "Work with HubSpot CRM data",
    "jira": "Work with Jira projects and issues",
    "linear": "Work with Linear projects and issues",
    "notion": "Read and update Notion workspaces",
    "salesforce": "Work with Salesforce CRM data",
    "shopify": "Work with Shopify stores",
    "slack": "Read and send Slack messages",
    "stripe": "Work with Stripe payments and customers",
    "zendesk": "Work with Zendesk support tickets",
}

PLATFORM_TOOLS: list[tuple[str, str]] = [
    (f"{PLATFORM_TOOL_PREFIX}{app}", _APP_DESCRIPTIONS[app]) for app in PLATFORM_APPS
]
