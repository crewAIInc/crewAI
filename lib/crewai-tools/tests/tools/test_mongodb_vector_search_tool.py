import json
from unittest.mock import patch

import pytest

from crewai_tools import MongoDBVectorSearchConfig, MongoDBVectorSearchTool


# Unit Test Fixtures
@pytest.fixture
def mongodb_vector_search_tool():
    tool = MongoDBVectorSearchTool(
        connection_string="foo", database_name="bar", collection_name="test"
    )
    tool._embed_texts = lambda x: [[0.1]]
    yield tool


# Unit Tests
def test_successful_query_execution(mongodb_vector_search_tool):
    # Enable embedding
    with patch.object(mongodb_vector_search_tool._coll, "aggregate") as mock_aggregate:
        mock_aggregate.return_value = [dict(text="foo", score=0.1, _id=1)]

        results = json.loads(mongodb_vector_search_tool._run(query="sandwiches"))

        assert len(results) == 1
        assert results[0]["text"] == "foo"
        assert results[0]["_id"] == 1


def test_provide_config():
    query_config = MongoDBVectorSearchConfig(limit=10)
    tool = MongoDBVectorSearchTool(
        connection_string="foo",
        database_name="bar",
        collection_name="test",
        query_config=query_config,
        vector_index_name="foo",
        embedding_model="bar",
    )
    tool._embed_texts = lambda x: [[0.1]]
    with patch.object(tool._coll, "aggregate") as mock_aggregate:
        mock_aggregate.return_value = [dict(text="foo", score=0.1, _id=1)]

        tool._run(query="sandwiches")
        assert mock_aggregate.mock_calls[-1].args[0][0]["$vectorSearch"]["limit"] == 10

        mock_aggregate.return_value = [dict(text="foo", score=0.1, _id=1)]


def test_cleanup_on_deletion(mongodb_vector_search_tool):
    with patch.object(mongodb_vector_search_tool, "_client") as mock_client:
        # Trigger cleanup
        mongodb_vector_search_tool.__del__()

        mock_client.close.assert_called_once()


def test_create_search_index(mongodb_vector_search_tool):
    with patch(
        "crewai_tools.tools.mongodb_vector_search_tool.vector_search.create_vector_search_index"
    ) as mock_create_search_index:
        mongodb_vector_search_tool.create_vector_search_index(dimensions=10)
        kwargs = mock_create_search_index.mock_calls[0].kwargs
        assert kwargs["dimensions"] == 10
        assert kwargs["similarity"] == "cosine"


def test_add_texts(mongodb_vector_search_tool):
    with patch.object(mongodb_vector_search_tool._coll, "bulk_write") as bulk_write:
        mongodb_vector_search_tool.add_texts(["foo"])
        args = bulk_write.mock_calls[0].args
        assert "ReplaceOne" in str(args[0][0])
        assert "foo" in str(args[0][0])
