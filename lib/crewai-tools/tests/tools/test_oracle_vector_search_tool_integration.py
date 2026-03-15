from __future__ import annotations

import json

import pytest

from crewai_tools import OracleVectorSearchTool
from tests.tools.oracle_db.conftest import has_oracle_vector_test_config

pytestmark = pytest.mark.filterwarnings(
    "ignore:datetime.datetime.utcnow\\(\\) is deprecated.*:DeprecationWarning:oci.base_client"
)


@pytest.mark.skipif(
    not has_oracle_vector_test_config(),
    reason="Oracle DB wallet or OCI live embedding config is not available",
)
def test_oracle_vector_search_tool_live(
    oracle_live_vector_tool_kwargs,
    oracle_vector_live_resources,
):
    tool = OracleVectorSearchTool(
        **(oracle_live_vector_tool_kwargs | oracle_vector_live_resources)
    )

    result = json.loads(tool._run("What is the refund policy?"))

    assert result["results"]
    top_result = result["results"][0]
    assert "refund policy" in top_result["content"].lower()
    assert top_result["metadata"]["category"] == "billing"
    assert top_result["metadata"]["topic"] == "billing"
    assert top_result["distance"] >= 0
