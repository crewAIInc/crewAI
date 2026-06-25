from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.k8s_agent_sandbox.settings import (
    K8sAgentSandboxToolSandboxSettings,
)
from crewai_tools.tools.k8s_agent_sandbox.lifecycle_manager import (
    EphemeralModeK8sAgentSandboxLifecycleManager,
    AttachModeK8sAgentSandboxLifecycleManager,
    PersistentModeK8sAgentSandboxLifecycleManager,
)
from crewai_tools.tools.k8s_agent_sandbox.toolset import K8sAgentSandboxToolset


@pytest.fixture
def mock_sandbox():
    sandbox = MagicMock()
    return sandbox


@pytest.fixture
def mock_sandbox_client():
    client = MagicMock()

    return client


@pytest.fixture
def mock_client_returns_mock_sandbox_in_create_sandbox(
    mock_sandbox, mock_sandbox_client
):
    mock_sandbox_client.create_sandbox.return_value = mock_sandbox


@pytest.fixture
def mock_client_returns_mock_sandbox_in_get_sandbox(mock_sandbox, mock_sandbox_client):
    mock_sandbox_client.get_sandbox.return_value = mock_sandbox


@pytest.fixture
def sample_sandbox_settings() -> K8sAgentSandboxToolSandboxSettings:
    return K8sAgentSandboxToolSandboxSettings(
        "my-warmpool",
        "my-namespace",
    )


@pytest.fixture
def mock_client_settings(mock_sandbox_client):
    settings = MagicMock()
    settings.client = mock_sandbox_client
    return settings


@pytest.fixture(params=["ephemeral", "attach", "persistent"])
def lifecycle_mode_name(request):
    return request.param


@pytest.fixture
def ephemeral_mode_lifecycle_manager(
    request, mock_client_settings, sample_sandbox_settings
):
    request.getfixturevalue("mock_client_returns_mock_sandbox_in_create_sandbox")
    return EphemeralModeK8sAgentSandboxLifecycleManager(
        mock_client_settings,
        sample_sandbox_settings,
    )


@pytest.fixture
def attach_mode_lifecycle_manager(
    request,
    mock_client_settings,
    sample_sandbox_settings,
):
    request.getfixturevalue("mock_client_returns_mock_sandbox_in_get_sandbox")
    return AttachModeK8sAgentSandboxLifecycleManager(
        mock_client_settings,
        sample_sandbox_settings,
        "my-claim",
    )


@pytest.fixture
def persistent_mode_lifecycle_manager(
    request, mock_client_settings, sample_sandbox_settings
):
    request.getfixturevalue("mock_client_returns_mock_sandbox_in_create_sandbox")
    request.getfixturevalue("mock_client_returns_mock_sandbox_in_get_sandbox")
    return PersistentModeK8sAgentSandboxLifecycleManager(
        mock_client_settings,
        sample_sandbox_settings,
    )


@pytest.fixture
def lifecycle_manager(request, lifecycle_mode_name):
    return request.getfixturevalue(f"{lifecycle_mode_name}_mode_lifecycle_manager")


@pytest.fixture
def sample_toolset(lifecycle_manager):
    toolset = K8sAgentSandboxToolset(
        lifecycle_manager=lifecycle_manager,
    )
    return toolset


@pytest.fixture
def lifecycle_mode_sandbox_termination_expected(lifecycle_manager):
    manager_type = type(lifecycle_manager)

    if manager_type is EphemeralModeK8sAgentSandboxLifecycleManager:
        return True
    else:
        return False
