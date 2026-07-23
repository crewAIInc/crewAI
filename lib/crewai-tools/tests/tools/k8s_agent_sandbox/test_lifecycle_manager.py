from typing import cast
from unittest.mock import MagicMock
from abc import ABC, abstractmethod

import pytest

from k8s_agent_sandbox.exceptions import SandboxNotFoundError

from crewai_tools.tools.k8s_agent_sandbox.lifecycle_manager import (
    K8sAgentSandboxLifecycleManager,
    EphemeralModeK8sAgentSandboxLifecycleManager,
    AttachModeK8sAgentSandboxLifecycleManager,
    PersistentModeK8sAgentSandboxLifecycleManager,
)


class LifecycleModeManagetTestBase(ABC):
    @abstractmethod
    def _create_manager(
        self,
        mock_client_settings,
        sample_sandbox_settings,
        **kwargs,
    ) -> K8sAgentSandboxLifecycleManager:
        pass

    @pytest.mark.usefixtures("mock_client_returns_mock_sandbox_in_create_sandbox")
    def test_close_acquired(self, mock_client_settings, sample_sandbox_settings, caplog):
        manager = self._create_manager(
            mock_client_settings,
            sample_sandbox_settings,
            # This has to make the manager close without releaseing a sandbox, otherwise the test case will hang
            close_timeout=0,
        )

        manager.acquire_sandbox()
        manager.close()

        assert "Closing anyway" in caplog.text


class TestEphemeralModeManager(LifecycleModeManagetTestBase):
    def _create_manager(
        self,
        mock_client_settings,
        sample_sandbox_settings,
        **kwargs,
    ):
        return EphemeralModeK8sAgentSandboxLifecycleManager(
            mock_client_settings,
            sample_sandbox_settings,
            **kwargs,
        )

    @pytest.fixture
    def manager(self, mock_client_settings, sample_sandbox_settings):
        return self._create_manager(mock_client_settings, sample_sandbox_settings)

    def test_main(self, manager, mock_sandbox_client):

        sandbox = manager.acquire_sandbox()
        assert mock_sandbox_client.create_sandbox.call_count == 1

        assert sandbox.terminate.call_count == 0
        manager.release_sandbox()
        assert sandbox.terminate.call_count == 1

        sandbox.reset_mock()

        sandbox = manager.acquire_sandbox()
        assert mock_sandbox_client.create_sandbox.call_count == 2

        assert not sandbox.terminate.called
        manager.release_sandbox()
        assert sandbox.terminate.call_count == 1

        assert mock_sandbox_client.get_sandbox.call_count == 0
        manager.close()
        assert sandbox.terminate.call_count == 1

    @pytest.mark.usefixtures("mock_client_returns_mock_sandbox_in_create_sandbox")
    def test_close_non_acquired(self, manager, mock_sandbox):
        assert mock_sandbox.terminate.call_count == 0
        manager.close()
        assert mock_sandbox.terminate.call_count == 0


class TestAttachModeManager(LifecycleModeManagetTestBase):
    def _create_manager(
        self,
        mock_client_settings,
        sample_sandbox_settings,
        **kwargs,
    ):
        return AttachModeK8sAgentSandboxLifecycleManager(
            mock_client_settings,
            sample_sandbox_settings,
            "my-claim",
            **kwargs,
        )

    @pytest.fixture
    def manager(self, mock_client_settings, sample_sandbox_settings):
        return self._create_manager(mock_client_settings, sample_sandbox_settings)

    @pytest.mark.usefixtures("mock_client_returns_mock_sandbox_in_create_sandbox")
    def test_acquire_and_release(
        self, manager, mock_sandbox_client, sample_sandbox_settings
    ):

        sandbox = manager.acquire_sandbox()
        assert not mock_sandbox_client.create_sandbox.called

        assert mock_sandbox_client.get_sandbox.call_count == 1

        assert mock_sandbox_client.get_sandbox.call_args.args[0] == "my-claim"
        assert (
            mock_sandbox_client.get_sandbox.call_args.kwargs["namespace"]
            == sample_sandbox_settings.namespace
        )

        manager.release_sandbox()
        assert not sandbox.terminate.called

        manager.close()
        assert not sandbox.terminate.called

    def test_non_existing_sandbox(self, manager, mock_sandbox_client):

        mock_sandbox_client.get_sandbox.side_effect = SandboxNotFoundError

        with pytest.raises(SandboxNotFoundError):
            manager.acquire_sandbox()

        manager.release_sandbox()
        manager.close()

    @pytest.mark.usefixtures("mock_client_returns_mock_sandbox_in_create_sandbox")
    def test_close_non_acquired(self, manager, mock_sandbox):
        assert mock_sandbox.terminate.call_count == 0
        manager.close()
        assert mock_sandbox.terminate.call_count == 0


class TestPersistentModeManager(LifecycleModeManagetTestBase):
    def _create_manager(
        self,
        mock_client_settings,
        sample_sandbox_settings,
        **kwargs,
    ):
        return PersistentModeK8sAgentSandboxLifecycleManager(
            mock_client_settings,
            sample_sandbox_settings,
            **kwargs,
        )
    @pytest.fixture
    def manager(self, mock_client_settings, sample_sandbox_settings):
        return self._create_manager(mock_client_settings, sample_sandbox_settings)

    def test_acquire_and_release(self, manager, mock_sandbox_client):
        sandbox = manager.acquire_sandbox()

        assert mock_sandbox_client.create_sandbox.call_count == 1

        manager.release_sandbox()

        assert not sandbox.terminate.called

        sandbox = manager.acquire_sandbox()

        assert mock_sandbox_client.create_sandbox.call_count == 1

        manager.release_sandbox()

        assert not sandbox.terminate.called

        manager.close()
        assert sandbox.terminate.called

    @pytest.mark.usefixtures("mock_client_returns_mock_sandbox_in_create_sandbox")
    def test_close_non_acquired(self, manager, mock_sandbox):
        assert mock_sandbox.terminate.call_count == 0
        manager.close()
        assert mock_sandbox.terminate.call_count == 0
