from __future__ import annotations

import json

import pytest

from crewai_tools import OracleHybridSearchTool
from tests.tools.oracle_db.conftest import has_oracle_hybrid_test_config


ORACLE_DB_AVAILABLE = True
try:
    import oracledb  # noqa: F401
except ImportError:
    ORACLE_DB_AVAILABLE = False


@pytest.mark.skipif(
    not ORACLE_DB_AVAILABLE or not has_oracle_hybrid_test_config(),
    reason="Requires oracledb plus a local Oracle wallet-backed DB configuration",
)
@pytest.mark.block_network(allowed_hosts=[r".*"])
def test_oracle_hybrid_search_tool_live_query(
    oracle_live_hybrid_tool_kwargs,
    oracle_hybrid_live_resources,
) -> None:
    tool = OracleHybridSearchTool(
        **(oracle_live_hybrid_tool_kwargs | oracle_hybrid_live_resources)
    )

    result = json.loads(tool._run("managed database"))

    assert "results" in result
    assert result["results"]
    assert "database" in result["results"][0]["content"].lower()
