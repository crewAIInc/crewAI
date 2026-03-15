from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from crewai_tools import (
    OCIGenAIInvokeAgentTool,
    OCIKnowledgeBaseTool,
    OCIObjectStorageReaderTool,
    OCIObjectStorageWriterTool,
)


def test_oci_object_storage_reader_tool_reads_text():
    client = MagicMock()
    client.get_namespace.return_value = SimpleNamespace(data="testns")
    client.get_object.return_value = SimpleNamespace(
        data=SimpleNamespace(content=b"oracle cloud")
    )

    tool = OCIObjectStorageReaderTool(client=client)

    result = tool._run("oci://my-bucket/docs/intro.txt")

    assert result == "oracle cloud"
    client.get_object.assert_called_once_with("testns", "my-bucket", "docs/intro.txt")


def test_oci_object_storage_writer_tool_writes_bytes():
    client = MagicMock()
    client.get_namespace.return_value = SimpleNamespace(data="testns")

    tool = OCIObjectStorageWriterTool(client=client)

    result = tool._run("oci://testns@my-bucket/reports/out.txt", "hello oci")

    assert result == "Successfully wrote content to oci://testns@my-bucket/reports/out.txt"
    client.put_object.assert_called_once_with(
        "testns",
        "my-bucket",
        "reports/out.txt",
        b"hello oci",
    )


def test_oci_genai_invoke_agent_tool_creates_session_and_chats():
    client = MagicMock()
    client.create_session.return_value = SimpleNamespace(
        data=SimpleNamespace(id="session-123")
    )
    client.chat.return_value = SimpleNamespace(
        data=SimpleNamespace(
            message=SimpleNamespace(
                content=SimpleNamespace(text="OCI agent response")
            )
        )
    )

    tool = OCIGenAIInvokeAgentTool(
        agent_endpoint_id="ocid1.genaiagentendpoint.oc1..example",
        client=client,
    )

    result = tool._run("Summarize Oracle Cloud Infrastructure")

    assert result == "OCI agent response"
    client.create_session.assert_called_once()
    client.chat.assert_called_once()
    assert client.chat.call_args.kwargs["agent_endpoint_id"] == (
        "ocid1.genaiagentendpoint.oc1..example"
    )


@patch("crewai_tools.tools.rag.rag_tool.build_embedder", return_value=MagicMock())
@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_oci_knowledge_base_tool_uses_oci_embedding_config(
    mock_create_client, _mock_build_embedder
):
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_create_client.return_value = mock_client

    tool = OCIKnowledgeBaseTool(config={"vectordb": {"provider": "chromadb", "config": {}}})

    assert tool.config["embedding_model"]["provider"] == "oci"
    assert (
        tool.config["embedding_model"]["config"]["model_name"]
        == "cohere.embed-english-v3.0"
    )
