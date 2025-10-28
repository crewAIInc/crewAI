from unittest.mock import MagicMock, patch

import pytest


# Mock the couchbase library before importing the tool
# This prevents ImportErrors if couchbase isn't installed in the test environment
mock_couchbase = MagicMock()
mock_couchbase.search = MagicMock()
mock_couchbase.cluster = MagicMock()
mock_couchbase.options = MagicMock()
mock_couchbase.vector_search = MagicMock()

# Simulate the structure needed for checks
mock_couchbase.cluster.Cluster = MagicMock()
mock_couchbase.options.SearchOptions = MagicMock()
mock_couchbase.vector_search.VectorQuery = MagicMock()
mock_couchbase.vector_search.VectorSearch = MagicMock()
mock_couchbase.search.SearchRequest = MagicMock()  # Mock the class itself
mock_couchbase.search.SearchRequest.create = MagicMock()  # Mock the class method


# Add necessary exception types if needed for testing error handling
class MockCouchbaseException(Exception):
    pass


mock_couchbase.exceptions = MagicMock()
mock_couchbase.exceptions.BucketNotFoundException = MockCouchbaseException
mock_couchbase.exceptions.ScopeNotFoundException = MockCouchbaseException
mock_couchbase.exceptions.CollectionNotFoundException = MockCouchbaseException
mock_couchbase.exceptions.IndexNotFoundException = MockCouchbaseException


import sys


sys.modules["couchbase"] = mock_couchbase
sys.modules["couchbase.search"] = mock_couchbase.search
sys.modules["couchbase.cluster"] = mock_couchbase.cluster
sys.modules["couchbase.options"] = mock_couchbase.options
sys.modules["couchbase.vector_search"] = mock_couchbase.vector_search
sys.modules["couchbase.exceptions"] = mock_couchbase.exceptions

# Now import the tool
from crewai_tools.tools.couchbase_tool.couchbase_tool import (
    CouchbaseFTSVectorSearchTool,
)


# --- Test Fixtures ---
@pytest.fixture(autouse=True)
def reset_global_mocks():
    """Reset call counts for globally defined mocks before each test."""
    # Reset the specific mock causing the issue
    mock_couchbase.vector_search.VectorQuery.reset_mock()
    # It's good practice to also reset other related global mocks
    # that might be called in your tests to prevent similar issues:
    mock_couchbase.vector_search.VectorSearch.from_vector_query.reset_mock()
    mock_couchbase.search.SearchRequest.create.reset_mock()


# Additional fixture to handle import pollution in full test suite
@pytest.fixture(autouse=True)
def ensure_couchbase_mocks():
    """Ensure that couchbase imports are properly mocked even when other tests have run first."""
    # This fixture ensures our mocks are in place regardless of import order
    original_modules = {}

    # Store any existing modules
    for module_name in [
        "couchbase",
        "couchbase.search",
        "couchbase.cluster",
        "couchbase.options",
        "couchbase.vector_search",
        "couchbase.exceptions",
    ]:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]

    # Ensure our mocks are active
    sys.modules["couchbase"] = mock_couchbase
    sys.modules["couchbase.search"] = mock_couchbase.search
    sys.modules["couchbase.cluster"] = mock_couchbase.cluster
    sys.modules["couchbase.options"] = mock_couchbase.options
    sys.modules["couchbase.vector_search"] = mock_couchbase.vector_search
    sys.modules["couchbase.exceptions"] = mock_couchbase.exceptions

    yield

    # Restore original modules if they existed
    for module_name, original_module in original_modules.items():
        if original_module is not None:
            sys.modules[module_name] = original_module


@pytest.fixture
def mock_cluster():
    cluster = MagicMock()
    bucket_manager = MagicMock()
    search_index_manager = MagicMock()
    bucket = MagicMock()
    scope = MagicMock()
    collection = MagicMock()
    scope_search_index_manager = MagicMock()

    # Setup mock return values for checks
    cluster.buckets.return_value = bucket_manager
    cluster.search_indexes.return_value = search_index_manager
    cluster.bucket.return_value = bucket
    bucket.scope.return_value = scope
    scope.collection.return_value = collection
    scope.search_indexes.return_value = scope_search_index_manager

    # Mock bucket existence check
    bucket_manager.get_bucket.return_value = True

    # Mock scope/collection existence check
    mock_scope_spec = MagicMock()
    mock_scope_spec.name = "test_scope"
    mock_collection_spec = MagicMock()
    mock_collection_spec.name = "test_collection"
    mock_scope_spec.collections = [mock_collection_spec]
    bucket.collections.return_value.get_all_scopes.return_value = [mock_scope_spec]

    # Mock index existence check
    mock_index_def = MagicMock()
    mock_index_def.name = "test_index"
    scope_search_index_manager.get_all_indexes.return_value = [mock_index_def]
    search_index_manager.get_all_indexes.return_value = [mock_index_def]

    return cluster


@pytest.fixture
def mock_embedding_function():
    # Simple mock embedding function
    # return lambda query: [0.1] * 10 # Example embedding vector
    return MagicMock(return_value=[0.1] * 10)


@pytest.fixture
def tool_config(mock_cluster, mock_embedding_function):
    return {
        "cluster": mock_cluster,
        "bucket_name": "test_bucket",
        "scope_name": "test_scope",
        "collection_name": "test_collection",
        "index_name": "test_index",
        "embedding_function": mock_embedding_function,
        "limit": 5,
        "embedding_key": "test_embedding",
        "scoped_index": True,
    }


@pytest.fixture
def couchbase_tool(tool_config):
    # Patch COUCHBASE_AVAILABLE to True for these tests
    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", True
    ):
        tool = CouchbaseFTSVectorSearchTool(**tool_config)
        return tool


@pytest.fixture
def mock_search_iter():
    mock_iter = MagicMock()
    # Simulate search results with a 'fields' attribute
    mock_row1 = MagicMock()
    mock_row1.fields = {"id": "doc1", "text": "content 1", "test_embedding": [0.1] * 10}
    mock_row2 = MagicMock()
    mock_row2.fields = {"id": "doc2", "text": "content 2", "test_embedding": [0.2] * 10}
    mock_iter.rows.return_value = [mock_row1, mock_row2]
    return mock_iter


# --- Test Cases ---


def test_initialization_success(couchbase_tool, tool_config):
    """Test successful initialization with valid config."""
    assert couchbase_tool.cluster == tool_config["cluster"]
    assert couchbase_tool.bucket_name == "test_bucket"
    assert couchbase_tool.scope_name == "test_scope"
    assert couchbase_tool.collection_name == "test_collection"
    assert couchbase_tool.index_name == "test_index"
    assert couchbase_tool.embedding_function is not None
    assert couchbase_tool.limit == 5
    assert couchbase_tool.embedding_key == "test_embedding"
    assert couchbase_tool.scoped_index

    # Check if helper methods were called during init (via mocks in fixture)
    couchbase_tool.cluster.buckets().get_bucket.assert_called_once_with("test_bucket")
    couchbase_tool.cluster.bucket().collections().get_all_scopes.assert_called_once()
    couchbase_tool.cluster.bucket().scope().search_indexes().get_all_indexes.assert_called_once()


def test_initialization_missing_required_args(mock_cluster, mock_embedding_function):
    """Test initialization fails when required arguments are missing."""
    base_config = {
        "cluster": mock_cluster,
        "bucket_name": "b",
        "scope_name": "s",
        "collection_name": "c",
        "index_name": "i",
        "embedding_function": mock_embedding_function,
    }
    required_keys = base_config.keys()

    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", True
    ):
        for key in required_keys:
            incomplete_config = base_config.copy()
            del incomplete_config[key]
            with pytest.raises(ValueError):
                CouchbaseFTSVectorSearchTool(**incomplete_config)


def test_initialization_couchbase_unavailable():
    """Test behavior when couchbase library is not available."""
    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", False
    ):
        with patch("click.confirm", return_value=False) as mock_confirm:
            with pytest.raises(
                ImportError, match="The 'couchbase' package is required"
            ):
                CouchbaseFTSVectorSearchTool(
                    cluster=MagicMock(),
                    bucket_name="b",
                    scope_name="s",
                    collection_name="c",
                    index_name="i",
                    embedding_function=MagicMock(),
                )
            mock_confirm.assert_called_once()  # Ensure user was prompted


def test_run_success_scoped_index(
    couchbase_tool, mock_search_iter, tool_config, mock_embedding_function
):
    """Test successful _run execution with a scoped index."""
    query = "find relevant documents"
    # expected_embedding = mock_embedding_function(query)

    # Mock the scope search method
    couchbase_tool._scope.search = MagicMock(return_value=mock_search_iter)
    # Mock the VectorQuery/VectorSearch/SearchRequest creation using runtime patching
    with (
        patch(
            "crewai_tools.tools.couchbase_tool.couchbase_tool.VectorQuery"
        ) as mock_vq,
        patch(
            "crewai_tools.tools.couchbase_tool.couchbase_tool.VectorSearch"
        ) as mock_vs,
        patch(
            "crewai_tools.tools.couchbase_tool.couchbase_tool.search.SearchRequest"
        ) as mock_sr,
        patch(
            "crewai_tools.tools.couchbase_tool.couchbase_tool.SearchOptions"
        ) as mock_so,
    ):
        # Set up the mock objects and their return values
        mock_vector_query = MagicMock()
        mock_vector_search = MagicMock()
        mock_search_req = MagicMock()
        mock_search_options = MagicMock()

        mock_vq.return_value = mock_vector_query
        mock_vs.from_vector_query.return_value = mock_vector_search
        mock_sr.create.return_value = mock_search_req
        mock_so.return_value = mock_search_options

        result = couchbase_tool._run(query=query)

        # Check embedding function call
        tool_config["embedding_function"].assert_called_once_with(query)

        # Check VectorQuery call
        mock_vq.assert_called_once_with(
            tool_config["embedding_key"],
            mock_embedding_function.return_value,
            tool_config["limit"],
        )
        # Check VectorSearch call
        mock_vs.from_vector_query.assert_called_once_with(mock_vector_query)
        # Check SearchRequest creation
        mock_sr.create.assert_called_once_with(mock_vector_search)
        # Check SearchOptions creation
        mock_so.assert_called_once_with(limit=tool_config["limit"], fields=["*"])

        # Check that scope search was called correctly
        couchbase_tool._scope.search.assert_called_once_with(
            tool_config["index_name"], mock_search_req, mock_search_options
        )

    # Check cluster search was NOT called
    couchbase_tool.cluster.search.assert_not_called()

    # Check result format (simple check for JSON structure)
    assert '"id": "doc1"' in result
    assert '"id": "doc2"' in result
    assert result.startswith("[")  # Should be valid JSON after concatenation


def test_run_success_global_index(
    tool_config, mock_search_iter, mock_embedding_function
):
    """Test successful _run execution with a global (non-scoped) index."""
    tool_config["scoped_index"] = False
    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", True
    ):
        couchbase_tool = CouchbaseFTSVectorSearchTool(**tool_config)

    query = "find global documents"
    # expected_embedding = mock_embedding_function(query)

    # Mock the cluster search method
    couchbase_tool.cluster.search = MagicMock(return_value=mock_search_iter)
    # Mock the VectorQuery/VectorSearch/SearchRequest creation using runtime patching
    with (
        patch(
            "crewai_tools.tools.couchbase_tool.couchbase_tool.VectorQuery"
        ) as mock_vq,
        patch(
            "crewai_tools.tools.couchbase_tool.couchbase_tool.VectorSearch"
        ) as mock_vs,
        patch(
            "crewai_tools.tools.couchbase_tool.couchbase_tool.search.SearchRequest"
        ) as mock_sr,
        patch(
            "crewai_tools.tools.couchbase_tool.couchbase_tool.SearchOptions"
        ) as mock_so,
    ):
        # Set up the mock objects and their return values
        mock_vector_query = MagicMock()
        mock_vector_search = MagicMock()
        mock_search_req = MagicMock()
        mock_search_options = MagicMock()

        mock_vq.return_value = mock_vector_query
        mock_vs.from_vector_query.return_value = mock_vector_search
        mock_sr.create.return_value = mock_search_req
        mock_so.return_value = mock_search_options

        result = couchbase_tool._run(query=query)

        # Check embedding function call
        tool_config["embedding_function"].assert_called_once_with(query)

        # Check VectorQuery/Search call
        mock_vq.assert_called_once_with(
            tool_config["embedding_key"],
            mock_embedding_function.return_value,
            tool_config["limit"],
        )
        mock_sr.create.assert_called_once_with(mock_vector_search)
        # Check SearchOptions creation
        mock_so.assert_called_once_with(limit=tool_config["limit"], fields=["*"])

        # Check that cluster search was called correctly
        couchbase_tool.cluster.search.assert_called_once_with(
            tool_config["index_name"], mock_search_req, mock_search_options
        )

    # Check scope search was NOT called
    couchbase_tool._scope.search.assert_not_called()

    # Check result format
    assert '"id": "doc1"' in result
    assert '"id": "doc2"' in result


def test_check_bucket_exists_fail(tool_config):
    """Test check for bucket non-existence."""
    mock_cluster = tool_config["cluster"]
    mock_cluster.buckets().get_bucket.side_effect = (
        mock_couchbase.exceptions.BucketNotFoundException("Bucket not found")
    )

    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", True
    ):
        with pytest.raises(ValueError, match="Bucket test_bucket does not exist."):
            CouchbaseFTSVectorSearchTool(**tool_config)


def test_check_scope_exists_fail(tool_config):
    """Test check for scope non-existence."""
    mock_cluster = tool_config["cluster"]
    # Simulate scope not being in the list returned
    mock_scope_spec = MagicMock()
    mock_scope_spec.name = "wrong_scope"
    mock_cluster.bucket().collections().get_all_scopes.return_value = [mock_scope_spec]

    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", True
    ):
        with pytest.raises(ValueError, match="Scope test_scope not found"):
            CouchbaseFTSVectorSearchTool(**tool_config)


def test_check_collection_exists_fail(tool_config):
    """Test check for collection non-existence."""
    mock_cluster = tool_config["cluster"]
    # Simulate collection not being in the scope's list
    mock_scope_spec = MagicMock()
    mock_scope_spec.name = "test_scope"
    mock_collection_spec = MagicMock()
    mock_collection_spec.name = "wrong_collection"
    mock_scope_spec.collections = [mock_collection_spec]  # Only has wrong collection
    mock_cluster.bucket().collections().get_all_scopes.return_value = [mock_scope_spec]

    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", True
    ):
        with pytest.raises(ValueError, match="Collection test_collection not found"):
            CouchbaseFTSVectorSearchTool(**tool_config)


def test_check_index_exists_fail_scoped(tool_config):
    """Test check for scoped index non-existence."""
    mock_cluster = tool_config["cluster"]
    # Simulate index not being in the list returned by scope manager
    mock_cluster.bucket().scope().search_indexes().get_all_indexes.return_value = []

    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", True
    ):
        with pytest.raises(ValueError, match="Index test_index does not exist"):
            CouchbaseFTSVectorSearchTool(**tool_config)


def test_check_index_exists_fail_global(tool_config):
    """Test check for global index non-existence."""
    tool_config["scoped_index"] = False
    mock_cluster = tool_config["cluster"]
    # Simulate index not being in the list returned by cluster manager
    mock_cluster.search_indexes().get_all_indexes.return_value = []

    with patch(
        "crewai_tools.tools.couchbase_tool.couchbase_tool.COUCHBASE_AVAILABLE", True
    ):
        with pytest.raises(ValueError, match="Index test_index does not exist"):
            CouchbaseFTSVectorSearchTool(**tool_config)
