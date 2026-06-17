from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.k8s_agent_sandbox.lifecycle_manager import (
    EphemeralModeK8sAgentSandboxLifecycleManager,
    AttachModeK8sAgentSandboxLifecycleManager,
    PersistentModeK8sAgentSandboxLifecycleManager,
)
from crewai_tools.tools.k8s_agent_sandbox.toolset import K8sAgentSandboxToolset


def test_add_tool(lifecycle_manager):
    toolset = K8sAgentSandboxToolset(
        lifecycle_manager,
    )

    assert len(toolset.tools) == 0

    tool = MagicMock()
    tool.name = "my_tool"

    toolset.add_tool(tool)

    assert len(toolset.tools) == 1

    with pytest.raises(ValueError, match="already in the toolset"):
        toolset.add_tool(tool)

    assert len(toolset.tools) == 1

    another_tool = MagicMock()
    another_tool.name = "another_tool"
    toolset.add_tool(another_tool)

    assert len(toolset.tools) == 2


class TestSandboxLifecycleModesSelection:
    @pytest.mark.parametrize(
        "extra_kwargs, expected_manager_class",
        [
            ({}, EphemeralModeK8sAgentSandboxLifecycleManager),
            (dict(claim_name="my_claim"), AttachModeK8sAgentSandboxLifecycleManager),
            (dict(persistent=True), PersistentModeK8sAgentSandboxLifecycleManager),
        ]
    )
    def test_lifecycle_modes(
        self,
        extra_kwargs,
        expected_manager_class,
        mock_client_settings,
        sample_sandbox_settings,
    ):
        toolset = K8sAgentSandboxToolset.create(
            sandbox_settings=sample_sandbox_settings,
            client_settings=mock_client_settings,
            **extra_kwargs,
        )

        assert type(toolset.lifecycle_manager) is expected_manager_class


    def test_persistent_and_attach_error(self, mock_client_settings, sample_sandbox_settings):
        with pytest.raises(ValueError, match="persistent and attach modes are mutually exclusive"):
          _ = K8sAgentSandboxToolset.create(
              sandbox_settings=sample_sandbox_settings,
              client_settings=mock_client_settings,
              persistent=True,
              claim_name="my_claim",
          )

