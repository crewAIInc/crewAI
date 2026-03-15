from __future__ import annotations

import json

import pytest

from crewai_tools import OracleTextSearchTool
from tests.tools.oracle_db.conftest import has_oracle_text_test_config


ORACLE_DB_AVAILABLE = True
try:
    import oracledb  # noqa: F401
except ImportError:
    ORACLE_DB_AVAILABLE = False


@pytest.mark.skipif(
    not ORACLE_DB_AVAILABLE or not has_oracle_text_test_config(),
    reason="Requires oracledb plus a local Oracle wallet-backed DB configuration",
)
@pytest.mark.block_network(allowed_hosts=[r".*"])
def test_oracle_text_search_tool_live_query(
    oracle_live_text_tool_kwargs,
    oracle_text_live_resources,
) -> None:
    tool = OracleTextSearchTool(**(oracle_live_text_tool_kwargs | oracle_text_live_resources))

    result = json.loads(tool._run("refund policy"))

    assert "results" in result
    assert result["results"]
    assert "refund" in result["results"][0]["content"].lower()
