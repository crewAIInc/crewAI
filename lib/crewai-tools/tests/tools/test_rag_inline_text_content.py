"""Inline text dicts must index the body, not the source label."""

import os
from unittest.mock import Mock, patch

from crewai_tools import RagTool


def test_inline_text_indexes_content():
    os.environ["OPENAI_API_KEY"] = "offline-test-key"
    client = Mock()

    with patch("crewai_tools.adapters.crewai_rag_adapter.create_client", return_value=client):
        tool = RagTool()
        tool.add(
            {
                "source": "batch-A",
                "content": "Solve 2x + 3 = 11.",
                "metadata": {"subject": "math"},
            },
            data_type="text",
        )

    document = client.add_documents.call_args.kwargs["documents"][0]
    assert document["content"] == "Solve 2x + 3 = 11."
    assert document["metadata"]["source"] == "batch-A"
