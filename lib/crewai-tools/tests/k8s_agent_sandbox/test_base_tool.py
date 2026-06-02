import sys
import pytest
from unittest.mock import MagicMock, patch

from crewai_tools.tools.k8s_agent_sandbox.base_tool import K8sBaseTool


class DummyK8sTool(K8sBaseTool):
    """A concrete subclass to allow instantiation of the BaseTool for testing."""
    def _run(self, *args, **kwargs) -> str:
        return "dummy"


@pytest.fixture(autouse=True)
def clean_class_state():
    """Ensure the SDK cache and atexit registry are reset between every test."""
    K8sBaseTool._sdk_cache.clear()
    yield
    K8sBaseTool._sdk_cache.clear()


@pytest.fixture
def mock_k8s_sdk():
    """Mocks the k8s_agent_sandbox SDK so we don't need it installed to run tests."""
    with patch.dict("sys.modules"):
        mock_sdk = MagicMock()
        mock_client_class = MagicMock()
        mock_sdk.SandboxClient = mock_client_class
        sys.modules["k8s_agent_sandbox"] = mock_sdk
        yield mock_client_class


def test_import_sandbox_client_success_and_caching(mock_k8s_sdk):
    """Test that the SDK is imported correctly and cached on subsequent calls."""
    # First call should import and cache
    client1 = K8sBaseTool._import_sandbox_client_class()
    assert client1 == mock_k8s_sdk
    assert "k8s_agent_sandbox.SandboxClient" in K8sBaseTool._sdk_cache

    # Second call should fetch from cache (we can verify by replacing the sys module)
    with patch.dict("sys.modules", {"k8s_agent_sandbox": None}):
        client2 = K8sBaseTool._import_sandbox_client_class()
        assert client2 == mock_k8s_sdk


def test_import_sandbox_client_missing_package():
    """Test that missing the SDK raises the custom, helpful ImportError."""
    with patch.dict("sys.modules"):
        # Force an ImportError if the code tries to import it
        sys.modules["k8s_agent_sandbox"] = None

        with pytest.raises(ImportError, match="The 'k8s_agent_sandbox' package is required"):
            K8sBaseTool._import_sandbox_client_class()


def test_get_sandbox_with_claim_name(mock_k8s_sdk):
    """Test that providing a claim_name bypasses creation and fetches the sandbox."""
    mock_sandbox = MagicMock()
    mock_client_instance = mock_k8s_sdk.return_value
    mock_client_instance.get_sandbox.return_value = mock_sandbox

    tool = DummyK8sTool(template="test-template", claim_name="my-claim")
    sandbox, should_kill = tool._get_sandbox()

    assert sandbox == mock_sandbox
    assert should_kill is False
    mock_client_instance.get_sandbox.assert_called_once_with("my-claim")
    mock_client_instance.create_sandbox.assert_not_called()


def test_get_sandbox_ephemeral_default(mock_k8s_sdk):
    """Test the default behavior: create a fresh sandbox and mark for termination."""
    mock_sandbox = MagicMock()
    mock_client_instance = mock_k8s_sdk.return_value
    mock_client_instance.create_sandbox.return_value = mock_sandbox

    tool = DummyK8sTool(template="test-template", namespace="custom-ns")
    sandbox, should_kill = tool._get_sandbox()

    assert sandbox == mock_sandbox
    assert should_kill is True
    mock_client_instance.create_sandbox.assert_called_once_with(
        template="test-template", namespace="custom-ns"
    )


@patch("atexit.register")
def test_get_sandbox_persistent(mock_atexit_register, mock_k8s_sdk):
    """Test persistent sandbox creation, caching, and atexit registration."""
    mock_sandbox = MagicMock()
    mock_client_instance = mock_k8s_sdk.return_value
    mock_client_instance.create_sandbox.return_value = mock_sandbox

    tool = DummyK8sTool(template="test-template", persistent=True)

    # First call: creates sandbox, registers atexit
    sandbox1, should_kill1 = tool._get_sandbox()
    assert sandbox1 == mock_sandbox
    assert should_kill1 is False
    mock_client_instance.create_sandbox.assert_called_once()
    mock_atexit_register.assert_called_once_with(tool.close)

    # Second call: fetches from local _persistent_sandbox cache
    sandbox2, should_kill2 = tool._get_sandbox()
    assert sandbox2 == mock_sandbox
    assert should_kill2 is False
    # Verify create_sandbox was NOT called a second time
    assert mock_client_instance.create_sandbox.call_count == 1


def test_release_sandbox_should_terminate():
    """Test that release_sandbox terminates when should_terminate is True."""
    tool = DummyK8sTool(template="test")
    mock_sandbox = MagicMock()

    tool._release_sandbox(mock_sandbox, should_terminate=True)
    mock_sandbox.terminate.assert_called_once()


def test_release_sandbox_should_not_terminate():
    """Test that release_sandbox skips termination when should_terminate is False."""
    tool = DummyK8sTool(template="test")
    mock_sandbox = MagicMock()

    tool._release_sandbox(mock_sandbox, should_terminate=False)
    mock_sandbox.terminate.assert_not_called()


@patch("logging.Logger.debug")
def test_release_sandbox_handles_exception(mock_log_debug):
    """Test that exceptions during termination are caught and logged."""
    tool = DummyK8sTool(template="test")
    mock_sandbox = MagicMock()
    mock_sandbox.terminate.side_effect = Exception("API Error")

    # Should not raise an error
    tool._release_sandbox(mock_sandbox, should_terminate=True)
    mock_sandbox.terminate.assert_called_once()
    mock_log_debug.assert_called_once()
    assert "Best-effort sandbox cleanup failed" in mock_log_debug.call_args[0][0]


def test_close_terminates_persistent_sandbox(mock_k8s_sdk):
    """Test that the close() method successfully terminates a persistent sandbox."""
    mock_sandbox = MagicMock()
    mock_k8s_sdk.return_value.create_sandbox.return_value = mock_sandbox

    tool = DummyK8sTool(template="test", persistent=True)
    tool._persistent_sandbox = mock_sandbox  # Simulate an active persistent sandbox

    tool.close()

    mock_sandbox.terminate.assert_called_once()
    assert tool._persistent_sandbox is None


@patch("logging.Logger.debug")
def test_close_handles_exception(mock_log_debug):
    """Test that exceptions during persistent sandbox cleanup are caught and logged."""
    tool = DummyK8sTool(template="test", persistent=True)
    mock_sandbox = MagicMock()
    mock_sandbox.terminate.side_effect = Exception("API Error")
    tool._persistent_sandbox = mock_sandbox

    tool.close()

    mock_sandbox.terminate.assert_called_once()
    assert tool._persistent_sandbox is None  # Should still nullify the cache
    mock_log_debug.assert_called_once()
    assert "Best-effort persistent sandbox cleanup failed" in mock_log_debug.call_args[0][0]
