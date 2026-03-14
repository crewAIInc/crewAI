from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DOCS_CONFIG_PATH = REPO_ROOT / "docs" / "docs.json"
INTEGRATION_OVERVIEW_PATH = (
    REPO_ROOT / "docs" / "en" / "tools" / "integration" / "overview.mdx"
)
TOOL_INTEGRATIONS_OVERVIEW_PATH = (
    REPO_ROOT / "docs" / "en" / "tools" / "tool-integrations" / "overview.mdx"
)
FIA_SIGNALS_DOC_PATH = (
    REPO_ROOT / "docs" / "en" / "tools" / "integration" / "fiasignalstools.mdx"
)
FIA_SIGNALS_ROUTE = '"en/tools/integration/fiasignalstools"'


def test_fia_signals_doc_page_exists() -> None:
    assert FIA_SIGNALS_DOC_PATH.exists()


def test_fia_signals_route_is_registered_in_docs_navigation() -> None:
    docs_config_text = DOCS_CONFIG_PATH.read_text(encoding="utf-8")
    assert docs_config_text.count(FIA_SIGNALS_ROUTE) >= 2


def test_fia_signals_route_is_linked_from_integration_overviews() -> None:
    integration_overview_text = INTEGRATION_OVERVIEW_PATH.read_text(encoding="utf-8")
    tool_integrations_overview_text = TOOL_INTEGRATIONS_OVERVIEW_PATH.read_text(
        encoding="utf-8"
    )
    expected_href = 'href="/en/tools/integration/fiasignalstools"'
    assert expected_href in integration_overview_text
    assert expected_href in tool_integrations_overview_text
