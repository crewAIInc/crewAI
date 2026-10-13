from crewai_core.platform_apps import (
    PLATFORM_CATALOG,
    PLATFORM_APP_CATEGORIES,
    PLATFORM_APP_DISPLAY_NAMES,
    PLATFORM_APP_TOOL_COUNTS,
    PLATFORM_APPLICATION_CATALOG,
    PLATFORM_APPS,
)


def test_platform_apps_contains_supported_application_catalog() -> None:
    assert isinstance(PLATFORM_CATALOG["applications"], list)
    assert len(PLATFORM_APPS) == 125
    assert len(set(PLATFORM_APPS)) == len(PLATFORM_APPS)
    assert PLATFORM_APPS[0] == "airtable"
    assert PLATFORM_APPS[-1] == "sap_s4hana"
    assert set(PLATFORM_APP_DISPLAY_NAMES) == set(PLATFORM_APPS)
    assert set(PLATFORM_APP_TOOL_COUNTS) == set(PLATFORM_APPS)
    assert PLATFORM_APP_DISPLAY_NAMES["github"] == "GitHub"
    assert PLATFORM_APP_TOOL_COUNTS["github"] == 877
    assert tuple(application.slug for application in PLATFORM_APPLICATION_CATALOG) == (
        PLATFORM_APPS
    )
    assert next(
        application for application in PLATFORM_APPLICATION_CATALOG if application.slug == "github"
    ).display_name == "GitHub"


def test_platform_app_categories_cover_the_catalog_once() -> None:
    categorized_apps = [
        app for category in PLATFORM_APP_CATEGORIES for app in category.apps
    ]

    assert len(PLATFORM_APP_CATEGORIES) == 12
    assert set(categorized_apps) == set(PLATFORM_APPS)
    assert len(categorized_apps) == len(set(categorized_apps))
    assert PLATFORM_APP_CATEGORIES[0].name == "Google Workspace & Google Cloud"
