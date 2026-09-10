from crewai.agents.agent_builder.base_agent import PLATFORM_APPS


def test_platform_apps_contains_supported_application_catalog() -> None:
    assert PLATFORM_APPS == (
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
    )
