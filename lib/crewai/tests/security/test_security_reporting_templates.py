from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
ISSUE_TEMPLATES_DIR = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    assert isinstance(data, dict), f"Expected YAML object in {path}"
    return data


def test_security_policy_is_in_supported_location() -> None:
    assert (REPO_ROOT / ".github" / "SECURITY.md").exists()


def test_security_policy_documents_escalation_window() -> None:
    policy = (REPO_ROOT / ".github" / "SECURITY.md").read_text(encoding="utf-8")

    assert "Acknowledgement" in policy
    assert "two business days" in policy
    assert "Escalation If You Do Not Hear Back" in policy
    assert "five business days" in policy


def test_issue_template_config_exposes_private_security_contact() -> None:
    config = _load_yaml(ISSUE_TEMPLATES_DIR / "config.yml")
    contact_links = config.get("contact_links")

    assert isinstance(contact_links, list)

    security_link = next(
        (
            link
            for link in contact_links
            if isinstance(link, dict)
            and "security" in str(link.get("name", "")).lower()
            and "advisories/new" in str(link.get("url", ""))
        ),
        None,
    )

    assert isinstance(security_link, dict)
    assert security_link["url"] == (
        "https://github.com/crewAIInc/crewAI/security/advisories/new"
    )
    assert "security@crewai.com" in str(security_link.get("about", ""))


def test_bug_report_template_requires_non_security_confirmation() -> None:
    bug_report = _load_yaml(ISSUE_TEMPLATES_DIR / "bug_report.yml")
    body = bug_report.get("body")

    assert isinstance(body, list)

    security_warning_present = any(
        isinstance(item, dict)
        and item.get("type") == "markdown"
        and "security vulnerability"
        in str((item.get("attributes") or {}).get("value", "")).lower()
        for item in body
    )
    assert security_warning_present

    confirmation_block = next(
        (
            item
            for item in body
            if isinstance(item, dict)
            and item.get("type") == "checkboxes"
            and item.get("id") == "non-security-confirmation"
        ),
        None,
    )

    assert isinstance(confirmation_block, dict)
    options = (confirmation_block.get("attributes") or {}).get("options")
    assert isinstance(options, list)
    assert options

    first_option = options[0]
    assert isinstance(first_option, dict)
    assert first_option.get("required") is True

    label = str(first_option.get("label", "")).lower()
    assert "vulnerability" in label
    assert "sensitive security information" in label
