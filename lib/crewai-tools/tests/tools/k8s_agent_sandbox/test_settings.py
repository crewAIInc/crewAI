from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock, patch
import time

from crewai_tools.tools.k8s_agent_sandbox.settings import (
    K8sAgentSandboxToolClientSettings,
)


class TestClientSettings:
    def test_client_is_built_once(self):
        settings = K8sAgentSandboxToolClientSettings()

        with patch(
            "crewai_tools.tools.k8s_agent_sandbox.settings.lazy_import_k8s_agent_sandbox"
        ) as lazy_import:
            client = settings.client

            assert settings.client is client
            assert lazy_import.return_value.SandboxClient.call_count == 1

    def test_concurrent_access_shares_one_client(self):
        settings = K8sAgentSandboxToolClientSettings()

        thread_count = 4
        start = Barrier(thread_count)
        clients = []

        def create_client(*args, **kwargs):
            # Widen the window a racing thread could slip through.
            time.sleep(0.01)
            return MagicMock()

        def get_client():
            start.wait()
            clients.append(settings.client)

        with patch(
            "crewai_tools.tools.k8s_agent_sandbox.settings.lazy_import_k8s_agent_sandbox"
        ) as lazy_import:
            lazy_import.return_value.SandboxClient.side_effect = create_client

            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                for future in [
                    executor.submit(get_client) for _ in range(thread_count)
                ]:
                    future.result()

            assert lazy_import.return_value.SandboxClient.call_count == 1

        assert len(clients) == thread_count
        assert all(client is clients[0] for client in clients)
