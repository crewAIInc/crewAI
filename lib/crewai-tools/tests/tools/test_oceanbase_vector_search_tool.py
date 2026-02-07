import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools import OceanBaseVectorSearchConfig


mock_pyobvector = MagicMock()
mock_pyobvector.ObVecClient = MagicMock()
mock_pyobvector.l2_distance = MagicMock(return_value="l2_func")
mock_pyobvector.cosine_distance = MagicMock(return_value="cosine_func")
mock_pyobvector.inner_product = MagicMock(return_value="ip_func")
sys.modules["pyobvector"] = mock_pyobvector


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 1536
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_client.embeddings.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_obvec_client():
    """Create a mock OceanBase vector client."""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def oceanbase_vector_search_tool(mock_openai_client, mock_obvec_client):
    """Create an OceanBaseVectorSearchTool with mocked clients."""
    from crewai_tools import OceanBaseVectorSearchTool

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with patch(
            "crewai_tools.tools.oceanbase_vector_search_tool.oceanbase_vector_search_tool.PYOBVECTOR_AVAILABLE",
            True,
        ):
            mock_pyobvector.ObVecClient.return_value = mock_obvec_client
            with patch("openai.Client") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                tool = OceanBaseVectorSearchTool(
                    connection_uri="127.0.0.1:2881",
                    user="root@test",
                    password="",
                    db_name="test",
                    table_name="test_table",
                )
                tool._openai_client = mock_openai_client
                tool._client = mock_obvec_client
                yield tool


def test_successful_query_execution(oceanbase_vector_search_tool, mock_obvec_client):
    """Test successful vector search query execution."""
    mock_obvec_client.ann_search.return_value = [
        ("test document content", {"source": "test.txt"}, 0.1),
        ("another document", {"source": "test2.txt"}, 0.2),
    ]

    results = json.loads(oceanbase_vector_search_tool._run(query="test query"))

    assert len(results) == 2
    assert results[0]["text"] == "test document content"
    assert results[0]["metadata"] == {"source": "test.txt"}
    assert results[0]["distance"] == 0.1


def test_query_with_custom_config(mock_openai_client, mock_obvec_client):
    """Test vector search with custom configuration."""
    from crewai_tools import OceanBaseVectorSearchTool

    query_config = OceanBaseVectorSearchConfig(
        limit=10,
        distance_func="cosine",
        distance_threshold=0.5,
    )

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with patch(
            "crewai_tools.tools.oceanbase_vector_search_tool.oceanbase_vector_search_tool.PYOBVECTOR_AVAILABLE",
            True,
        ):
            mock_pyobvector.ObVecClient.return_value = mock_obvec_client
            with patch("openai.Client") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                tool = OceanBaseVectorSearchTool(
                    connection_uri="127.0.0.1:2881",
                    user="root@test",
                    db_name="test",
                    table_name="test_table",
                    query_config=query_config,
                )
                tool._openai_client = mock_openai_client
                tool._client = mock_obvec_client

    mock_obvec_client.ann_search.return_value = [("doc", {}, 0.3)]

    tool._run(query="test")

    call_kwargs = mock_obvec_client.ann_search.call_args.kwargs
    assert call_kwargs["topk"] == 10
    assert call_kwargs["distance_threshold"] == 0.5


def test_add_texts(oceanbase_vector_search_tool, mock_obvec_client):
    """Test adding texts to the OceanBase table."""
    texts = ["document 1", "document 2"]
    metadatas = [{"source": "file1.txt"}, {"source": "file2.txt"}]

    result_ids = oceanbase_vector_search_tool.add_texts(texts, metadatas=metadatas)

    assert len(result_ids) == 2
    mock_obvec_client.insert.assert_called_once()
    call_args = mock_obvec_client.insert.call_args
    assert call_args[0][0] == "test_table"
    assert len(call_args[1]["data"]) == 2


def test_add_texts_without_metadata(oceanbase_vector_search_tool, mock_obvec_client):
    """Test adding texts without metadata."""
    texts = ["document 1", "document 2"]

    result_ids = oceanbase_vector_search_tool.add_texts(texts)

    assert len(result_ids) == 2
    mock_obvec_client.insert.assert_called_once()


def test_error_handling(oceanbase_vector_search_tool, mock_obvec_client):
    """Test error handling during search."""
    mock_obvec_client.ann_search.side_effect = Exception("Database connection error")

    result = json.loads(oceanbase_vector_search_tool._run(query="test"))

    assert "error" in result
    assert "Database connection error" in result["error"]


def test_config_defaults():
    """Test OceanBaseVectorSearchConfig default values."""
    config = OceanBaseVectorSearchConfig()

    assert config.limit == 4
    assert config.distance_func == "l2"
    assert config.distance_threshold is None
    assert config.include_embeddings is False


def test_config_custom_values():
    """Test OceanBaseVectorSearchConfig with custom values."""
    config = OceanBaseVectorSearchConfig(
        limit=20,
        distance_func="cosine",
        distance_threshold=0.8,
        include_embeddings=True,
    )

    assert config.limit == 20
    assert config.distance_func == "cosine"
    assert config.distance_threshold == 0.8
    assert config.include_embeddings is True


def test_tool_schema():
    """Test OceanBaseToolSchema validation."""
    from crewai_tools import OceanBaseToolSchema

    schema = OceanBaseToolSchema(query="test query")
    assert schema.query == "test query"


def test_tool_schema_requires_query():
    """Test that OceanBaseToolSchema requires a query."""
    from crewai_tools import OceanBaseToolSchema
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OceanBaseToolSchema()


def test_distance_function_selection(oceanbase_vector_search_tool):
    """Test that the correct distance function is selected."""
    oceanbase_vector_search_tool.query_config = OceanBaseVectorSearchConfig(
        distance_func="l2"
    )
    func = oceanbase_vector_search_tool._get_distance_func()
    assert func == mock_pyobvector.l2_distance

    oceanbase_vector_search_tool.query_config = OceanBaseVectorSearchConfig(
        distance_func="cosine"
    )
    func = oceanbase_vector_search_tool._get_distance_func()
    assert func == mock_pyobvector.cosine_distance

    oceanbase_vector_search_tool.query_config = OceanBaseVectorSearchConfig(
        distance_func="inner_product"
    )
    func = oceanbase_vector_search_tool._get_distance_func()
    assert func == mock_pyobvector.inner_product
