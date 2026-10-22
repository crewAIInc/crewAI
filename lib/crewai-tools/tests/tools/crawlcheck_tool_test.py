import json
from unittest.mock import patch

import pytest

from crewai_tools import CrawlCheckSiteCheckTool, CrawlCheckVerifyFactTool


def _fake_tools(cc, allow_warn=False, **kwargs):
    return [
        {
            "name": "crawlcheck_check_site",
            "execute": lambda args: {
                "domain": args["site"],
                "action": args["action"],
                "proceed": args["site"] != "blocked.example",
                "decision": "allow",
            },
        },
        {
            "name": "crawlcheck_verify_fact",
            "execute": lambda args: {
                "ok": True,
                "verified": args["value"] == "303-555-0100",
                "field": args["field"],
            },
        },
    ]


@pytest.fixture
def fake_sdk():
    with (
        patch("crawlcheck.agent_tools", side_effect=_fake_tools),
        patch("crawlcheck.CrawlCheck"),
    ):
        yield


def test_site_check_proceeds(fake_sdk):
    tool = CrawlCheckSiteCheckTool()
    out = json.loads(tool._run(site="example.com", action="cite"))
    assert out["proceed"] is True and out["action"] == "cite"


def test_site_check_blocked_and_unknown_action(fake_sdk):
    tool = CrawlCheckSiteCheckTool()
    out = json.loads(tool._run(site="blocked.example", action="dance"))
    assert out["proceed"] is False and out["action"] == "read"


def test_verify_fact(fake_sdk):
    tool = CrawlCheckVerifyFactTool()
    assert (
        json.loads(tool._run(site="example.com", field="phone", value="303-555-0100"))[
            "verified"
        ]
        is True
    )
    bad = json.loads(tool._run(site="example.com", field="colour", value="blue"))
    assert bad["ok"] is False


def test_schema_fields(fake_sdk):
    assert set(CrawlCheckSiteCheckTool().args_schema.model_fields) == {"site", "action"}
    assert set(CrawlCheckVerifyFactTool().args_schema.model_fields) == {
        "site",
        "field",
        "value",
    }
