from __future__ import annotations

import os
import uuid

import pytest

from crewai_tools import OCIKnowledgeBaseTool


OCI_SDK_AVAILABLE = True
try:
    import oci  # noqa: F401
except ImportError:
    OCI_SDK_AVAILABLE = False


def _has_oci_test_config() -> bool:
    return bool(
        os.getenv("OCI_COMPARTMENT_ID")
        and (os.getenv("OCI_SERVICE_ENDPOINT") or os.getenv("OCI_REGION"))
    )


@pytest.mark.skipif(
    not OCI_SDK_AVAILABLE or not _has_oci_test_config(),
    reason="Requires OCI SDK plus OCI_COMPARTMENT_ID and OCI endpoint configuration",
)
@pytest.mark.block_network(allowed_hosts=[r".*"])
def test_oci_knowledge_base_tool_live_query(tmp_path) -> None:
    document_path = tmp_path / "oracle_notes.txt"
    document_path.write_text(
        (
            "Oracle Cloud Infrastructure includes Autonomous Database. "
            "Autonomous Database is a managed Oracle database service."
        ),
        encoding="utf-8",
    )

    tool_kwargs = {
        "knowledge_source": str(document_path),
        "collection_name": f"oci_kb_live_test_{uuid.uuid4().hex[:8]}",
        "compartment_id": os.getenv("OCI_COMPARTMENT_ID"),
        "auth_type": os.getenv("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.getenv("OCI_AUTH_PROFILE", "DEFAULT"),
        "auth_file_location": os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config"),
    }
    if os.getenv("OCI_REGION"):
        tool_kwargs["region"] = os.getenv("OCI_REGION")
    if os.getenv("OCI_SERVICE_ENDPOINT"):
        tool_kwargs["service_endpoint"] = os.getenv("OCI_SERVICE_ENDPOINT")

    tool = OCIKnowledgeBaseTool(**tool_kwargs)

    result = tool._run("Which Oracle service is described in the document?")

    assert "Autonomous Database" in result
