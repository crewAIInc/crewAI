import inspect
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from crewai_tools import CrawlCheckSiteCheckTool, CrawlCheckVerifyFactTool


def _fake_tools(cc, agent=None, allow_warn=False, confirm=None, max_chars=20000):
    """Same signature and return shape as crawlcheck.agent_tools (crawlcheck>=1.7.0)."""
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
    """Put a stand-in `crawlcheck` module in sys.modules so the tests run without the optional extra."""
    fake = ModuleType("crawlcheck")
    fake.CrawlCheck = MagicMock(name="CrawlCheck")
    fake.agent_tools = MagicMock(name="agent_tools", side_effect=_fake_tools)
    original = sys.modules.get("crawlcheck")
    sys.modules["crawlcheck"] = fake
    try:
        yield fake
    finally:
        if original is not None:
            sys.modules["crawlcheck"] = original
        else:
            sys.modules.pop("crawlcheck", None)


def test_site_check_proceeds(fake_sdk):
    tool = CrawlCheckSiteCheckTool()
    out = json.loads(tool._run(site="example.com", action="cite"))
    assert out["proceed"] is True and out["action"] == "cite"


def test_site_check_blocked(fake_sdk):
    tool = CrawlCheckSiteCheckTool()
    out = json.loads(tool._run(site="blocked.example", action="read"))
    assert out["proceed"] is False


def test_unknown_action_fails_closed(fake_sdk):
    tool = CrawlCheckSiteCheckTool()
    out = json.loads(tool._run(site="example.com", action="pay"))
    assert out["proceed"] is False and out["ok"] is False
    # case and whitespace are normalised, not downgraded to "read"
    out = json.loads(tool._run(site="example.com", action=" Transact "))
    assert out["action"] == "transact"


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


def test_schema_rejects_unknown_action_and_field(fake_sdk):
    with pytest.raises(ValueError, match="arguments validation failed"):
        CrawlCheckSiteCheckTool().run(site="example.com", action="pay")
    with pytest.raises(ValueError, match="arguments validation failed"):
        CrawlCheckVerifyFactTool().run(site="example.com", field="colour", value="blue")


def test_allow_warn_is_passed_to_the_sdk(fake_sdk):
    CrawlCheckSiteCheckTool(allow_warn=True)
    assert fake_sdk.agent_tools.call_args.kwargs == {"allow_warn": True}


def test_wrapper_matches_the_published_sdk():
    """Contract test against the real package; skipped when the extra is not installed.

    The wrapper relies on agent_tools(cc, allow_warn=...) returning dicts with `name` and a
    callable `execute`. The mocked tests above cannot catch a change to that, so this one reads
    the installed SDK. Building the tools makes no network request.
    """
    crawlcheck = pytest.importorskip("crawlcheck")
    assert "allow_warn" in inspect.signature(crawlcheck.agent_tools).parameters
    tools = {
        t["name"]: t
        for t in crawlcheck.agent_tools(crawlcheck.CrawlCheck(), allow_warn=True)
    }
    for name in ("crawlcheck_check_site", "crawlcheck_verify_fact"):
        assert callable(tools[name]["execute"])
    # An unknown action is refused by the SDK itself, before any request is made.
    refused = tools["crawlcheck_check_site"]["execute"](
        {"site": "example.com", "action": "pay"}
    )
    assert refused["proceed"] is False
