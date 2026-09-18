from crewai_core.platform_apps import PLATFORM_APPS


def test_platform_apps_contains_supported_application_catalog() -> None:
    assert PLATFORM_APPS == (
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
    )
