"""CrewAI Platform application catalog."""

from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Final, TypeAlias


PlatformApp: TypeAlias = str


@dataclass(frozen=True)
class PlatformToolDefinition:
    """Describe one selectable CrewAI Platform action."""

    slug: str
    display_name: str


@dataclass(frozen=True)
class PlatformApplicationDefinition:
    """Describe one CrewAI Platform application and its available actions."""

    slug: PlatformApp
    display_name: str
    tools: tuple[PlatformToolDefinition, ...]


@dataclass(frozen=True)
class PlatformAppCategory:
    """Describe a presentation group of CrewAI Platform applications."""

    name: str
    apps: tuple[PlatformApp, ...]


def _load_platform_catalog() -> tuple[PlatformApplicationDefinition, ...]:
    """Load and validate the generated Clipper application catalog."""
    catalog_path = files("crewai_core").joinpath("platform_catalog.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    applications = catalog.get("applications") if isinstance(catalog, dict) else None
    if not isinstance(applications, list):
        raise ValueError("Platform catalog must contain an 'applications' list.")

    applications_by_slug: set[str] = set()
    parsed_applications: list[PlatformApplicationDefinition] = []
    for application in applications:
        if not isinstance(application, dict):
            raise ValueError("Every platform catalog application must be an object.")
        slug = application.get("slug")
        display_name = application.get("display_name")
        if not isinstance(slug, str) or not slug:
            raise ValueError("Every platform catalog application must have a slug.")
        if not isinstance(display_name, str) or not display_name:
            raise ValueError(
                f"Platform catalog application '{slug}' must have a display name."
            )
        if slug in applications_by_slug:
            raise ValueError(f"Platform catalog contains duplicate slug '{slug}'.")
        tools = application.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError(
                f"Platform catalog application '{slug}' must have a tools list."
            )
        parsed_tools: list[PlatformToolDefinition] = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError(
                    f"Platform catalog application '{slug}' contains an invalid tool."
                )
            tool_slug = tool.get("slug")
            tool_display_name = tool.get("display_name")
            if not isinstance(tool_slug, str) or not tool_slug:
                raise ValueError(
                    f"Platform catalog application '{slug}' contains a tool without a slug."
                )
            if not isinstance(tool_display_name, str) or not tool_display_name:
                raise ValueError(
                    f"Platform catalog tool '{slug}/{tool_slug}' must have a display name."
                )
            parsed_tools.append(
                PlatformToolDefinition(
                    slug=tool_slug,
                    display_name=tool_display_name,
                )
            )
        applications_by_slug.add(slug)
        parsed_applications.append(
            PlatformApplicationDefinition(
                slug=slug,
                display_name=display_name,
                tools=tuple(parsed_tools),
            )
        )

    return tuple(parsed_applications)


PLATFORM_APPLICATION_CATALOG: Final[tuple[PlatformApplicationDefinition, ...]] = (
    _load_platform_catalog()
)
PLATFORM_APPS: Final[tuple[str, ...]] = tuple(
    application.slug for application in PLATFORM_APPLICATION_CATALOG
)
PLATFORM_APP_DISPLAY_NAMES: Final[dict[str, str]] = {
    application.slug: application.display_name
    for application in PLATFORM_APPLICATION_CATALOG
}
PLATFORM_APP_TOOL_COUNTS: Final[dict[str, int]] = {
    application.slug: len(application.tools)
    for application in PLATFORM_APPLICATION_CATALOG
}
PLATFORM_APP_TOOLS: Final[dict[str, tuple[PlatformToolDefinition, ...]]] = {
    application.slug: application.tools for application in PLATFORM_APPLICATION_CATALOG
}


def _platform_app_categories() -> tuple[PlatformAppCategory, ...]:
    """Return the curated application groups used by integration pickers."""
    return (
        PlatformAppCategory(
            "Google Workspace & Google Cloud",
            (
                "gmail",
                "google_calendar",
                "google_docs",
                "google_drive",
                "google_sheets",
                "google_slides",
                "google_contacts",
                "google_classroom",
                "googlemeet",
                "googlephotos",
                "googletasks",
                "googlesuper",
                "google_maps",
                "google_search_console",
                "google_analytics",
                "googleads",
                "googlebigquery",
            ),
        ),
        PlatformAppCategory(
            "Microsoft 365",
            (
                "microsoft_excel",
                "microsoft_onedrive",
                "microsoft_outlook",
                "microsoft_sharepoint",
                "microsoft_teams",
                "microsoft_word",
            ),
        ),
        PlatformAppCategory(
            "Project & collaboration",
            (
                "airtable",
                "asana",
                "basecamp",
                "clickup",
                "confluence",
                "dart",
                "jira",
                "linear",
                "monday",
                "notion",
                "productboard",
                "todoist",
                "trello",
                "ticktick",
                "wrike",
            ),
        ),
        PlatformAppCategory(
            "Design & content",
            ("canva", "contentful", "figma", "miro", "mural", "zeplin"),
        ),
        PlatformAppCategory(
            "Communication, meetings & scheduling",
            (
                "cal",
                "calendly",
                "dialpad",
                "discord",
                "discordbot",
                "fathom",
                "gong",
                "granola_mcp",
                "slack",
                "slackbot",
                "whatsapp",
                "zoom",
                "zoho_mail",
            ),
        ),
        PlatformAppCategory(
            "CRM, sales & customer support",
            (
                "attio",
                "dynamics365",
                "gorgias",
                "hubspot",
                "intercom",
                "pylon_mcp",
                "salesforce",
                "zendesk",
                "zoho",
                "zoho_bigin",
                "zoho_desk",
            ),
        ),
        PlatformAppCategory(
            "Marketing, social & audience",
            (
                "dub",
                "eventbrite",
                "facebook",
                "instagram",
                "kit",
                "linkedin",
                "mailchimp",
                "omnisend",
                "pinterest",
                "pinterest_ads",
                "reddit",
                "reddit_ads",
                "tiktok_ads",
                "toneden",
                "twitch",
                "typeform",
                "youtube",
            ),
        ),
        PlatformAppCategory(
            "Finance & commerce",
            (
                "freshbooks",
                "gumroad",
                "mercury_mcp",
                "moneybird",
                "quickbooks",
                "shippo",
                "shopify",
                "square",
                "splitwise",
                "stripe",
                "ynab",
                "zoho_books",
            ),
        ),
        PlatformAppCategory(
            "Business operations & HR",
            ("apaleo", "greenhouse", "harvest", "sap_s4hana", "servicem8", "timely"),
        ),
        PlatformAppCategory(
            "Engineering, data & infrastructure",
            (
                "bitbucket",
                "daytona",
                "digital_ocean",
                "github",
                "gitlab",
                "hugging_face",
                "pagerduty",
                "prisma",
                "sentry",
                "supabase",
                "wakatime",
            ),
        ),
        PlatformAppCategory(
            "Files, documents & knowledge",
            ("boldsign", "box", "dropbox", "linkhut", "notebook_lm", "roam"),
        ),
        PlatformAppCategory(
            "Personal, utilities & reference",
            ("exist", "pushbullet", "stack_exchange", "ticketmaster", "yandex"),
        ),
    )


def _validate_platform_app_categories(
    categories: tuple[PlatformAppCategory, ...], apps: tuple[PlatformApp, ...]
) -> tuple[PlatformAppCategory, ...]:
    """Ensure every catalog application belongs to exactly one category."""
    categorized_apps = [app for category in categories for app in category.apps]
    duplicates = {app for app in categorized_apps if categorized_apps.count(app) > 1}
    if duplicates:
        raise ValueError(
            "Platform app categories contain duplicate applications: "
            f"{', '.join(sorted(duplicates))}."
        )

    categorized_app_set = set(categorized_apps)
    app_set = set(apps)
    unknown = categorized_app_set - app_set
    if unknown:
        raise ValueError(
            "Platform app categories contain unknown applications: "
            f"{', '.join(sorted(unknown))}."
        )
    missing = app_set - categorized_app_set
    if missing:
        raise ValueError(
            "Platform app categories are missing applications: "
            f"{', '.join(sorted(missing))}."
        )
    return categories


PLATFORM_APP_CATEGORIES: Final[tuple[PlatformAppCategory, ...]] = (
    _validate_platform_app_categories(_platform_app_categories(), PLATFORM_APPS)
)
