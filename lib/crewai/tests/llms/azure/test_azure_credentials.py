"""Tests for Azure credential resolution chain in AzureCompletion.

Covers the four credential paths:
1. WorkloadIdentityCredential (OIDC federation)
2. ClientSecretCredential (Service Principal)
3. DefaultAzureCredential (Managed Identity / CLI fallback)
4. AzureKeyCredential (API key - existing path)
"""

from unittest.mock import patch, MagicMock

import pytest


# Use a non-Azure-OpenAI endpoint to avoid _validate_and_fix_endpoint suffixing
ENDPOINT = "https://test-ai.services.example.com"


@pytest.fixture
def _clear_azure_env(monkeypatch):
    """Remove all Azure env vars to start clean."""
    for key in [
        "AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT",
        "AZURE_API_BASE", "AZURE_API_VERSION", "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_FEDERATED_TOKEN_FILE",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.usefixtures("_clear_azure_env")
class TestCredentialResolution:
    """Tests for AzureCompletion._resolve_credential."""

    def test_api_key_credential_when_api_key_set(self):
        """Path 4: API key produces AzureKeyCredential."""
        from crewai.llms.providers.azure.completion import AzureCompletion
        from azure.core.credentials import AzureKeyCredential

        completion = AzureCompletion(
            model="gpt-4",
            api_key="test-key",
            endpoint=ENDPOINT,
        )
        cred = completion._resolve_credential()
        assert isinstance(cred, AzureKeyCredential)

    def test_api_key_from_env(self, monkeypatch):
        """Path 4: api_key picked up from AZURE_API_KEY env var."""
        from crewai.llms.providers.azure.completion import AzureCompletion
        from azure.core.credentials import AzureKeyCredential

        monkeypatch.setenv("AZURE_API_KEY", "env-key")
        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        completion = AzureCompletion(model="gpt-4")
        cred = completion._resolve_credential()
        assert isinstance(cred, AzureKeyCredential)

    def test_workload_identity_credential(self, monkeypatch, tmp_path):
        """Path 1: OIDC federation via WorkloadIdentityCredential."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        token_file = tmp_path / "token.txt"
        token_file.write_text("eyJhbGciOiJSUzI1NiJ9.test")

        monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        mock_wi_cred = MagicMock()
        with patch(
            "azure.identity.WorkloadIdentityCredential",
            return_value=mock_wi_cred,
        ) as mock_cls:
            completion = AzureCompletion(
                model="gpt-4",
                azure_tenant_id="tenant-123",
                azure_client_id="client-456",
            )
            cred = completion._resolve_credential()
            assert cred is mock_wi_cred
            # Called at least once with the right args (init may also call it)
            mock_cls.assert_any_call(
                tenant_id="tenant-123",
                client_id="client-456",
                token_file_path=str(token_file),
            )

    def test_workload_identity_from_env_vars(self, monkeypatch, tmp_path):
        """Path 1: All WI fields discovered from environment."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        token_file = tmp_path / "token.txt"
        token_file.write_text("eyJhbGciOiJSUzI1NiJ9.test")

        monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("AZURE_TENANT_ID", "env-tenant")
        monkeypatch.setenv("AZURE_CLIENT_ID", "env-client")
        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        mock_wi_cred = MagicMock()
        with patch(
            "azure.identity.WorkloadIdentityCredential",
            return_value=mock_wi_cred,
        ) as mock_cls:
            completion = AzureCompletion(model="gpt-4")
            cred = completion._resolve_credential()
            assert cred is mock_wi_cred
            mock_cls.assert_any_call(
                tenant_id="env-tenant",
                client_id="env-client",
                token_file_path=str(token_file),
            )

    def test_client_secret_credential(self, monkeypatch):
        """Path 2: Service Principal with client secret."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        monkeypatch.setenv("AZURE_CLIENT_SECRET", "sp-secret")
        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        mock_cs_cred = MagicMock()
        with patch(
            "azure.identity.ClientSecretCredential",
            return_value=mock_cs_cred,
        ) as mock_cls:
            completion = AzureCompletion(
                model="gpt-4",
                azure_tenant_id="tenant-123",
                azure_client_id="client-456",
            )
            cred = completion._resolve_credential()
            assert cred is mock_cs_cred
            mock_cls.assert_any_call(
                tenant_id="tenant-123",
                client_id="client-456",
                client_secret="sp-secret",
            )

    def test_default_azure_credential_when_no_api_key(self, monkeypatch):
        """Path 3: DefaultAzureCredential when no api_key and no SP/WI vars."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        mock_default_cred = MagicMock()
        with patch(
            "azure.identity.DefaultAzureCredential",
            return_value=mock_default_cred,
        ):
            completion = AzureCompletion(model="gpt-4")
            cred = completion._resolve_credential()
            assert cred is mock_default_cred

    def test_workload_identity_takes_priority_over_api_key(self, monkeypatch, tmp_path):
        """WI credential should take priority even when api_key is also set."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        token_file = tmp_path / "token.txt"
        token_file.write_text("eyJhbGciOiJSUzI1NiJ9.test")

        monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("AZURE_API_KEY", "should-not-use-this")
        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        mock_wi_cred = MagicMock()
        with patch(
            "azure.identity.WorkloadIdentityCredential",
            return_value=mock_wi_cred,
        ):
            completion = AzureCompletion(
                model="gpt-4",
                azure_tenant_id="tenant-123",
                azure_client_id="client-456",
            )
            cred = completion._resolve_credential()
            assert cred is mock_wi_cred

    def test_client_secret_takes_priority_over_api_key(self, monkeypatch):
        """SP credential should take priority over API key."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        monkeypatch.setenv("AZURE_CLIENT_SECRET", "sp-secret")
        monkeypatch.setenv("AZURE_API_KEY", "should-not-use-this")
        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        mock_cs_cred = MagicMock()
        with patch(
            "azure.identity.ClientSecretCredential",
            return_value=mock_cs_cred,
        ):
            completion = AzureCompletion(
                model="gpt-4",
                azure_tenant_id="tenant-123",
                azure_client_id="client-456",
            )
            cred = completion._resolve_credential()
            assert cred is mock_cs_cred

    def test_raises_when_no_api_key_and_no_azure_identity(self, monkeypatch):
        """ValueError when no api_key and azure-identity not installed."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        with patch.dict("sys.modules", {"azure.identity": None}):
            completion = AzureCompletion(model="gpt-4")
            with pytest.raises(ValueError, match="Azure API key is required"):
                completion._resolve_credential()

    def test_endpoint_still_required(self, monkeypatch, tmp_path):
        """Endpoint is always required regardless of credential type."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        token_file = tmp_path / "token.txt"
        token_file.write_text("test-jwt")

        monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("AZURE_TENANT_ID", "tenant-123")
        monkeypatch.setenv("AZURE_CLIENT_ID", "client-456")

        completion = AzureCompletion(model="gpt-4")
        with pytest.raises(ValueError, match="Azure endpoint is required"):
            completion._make_client_kwargs()

    def test_deferred_build_picks_up_wi_env_vars(self, monkeypatch, tmp_path):
        """Env vars set after construction are picked up on deferred build."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        # Construct with endpoint only — no credentials yet
        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)
        completion = AzureCompletion(model="gpt-4")

        # Now set WI env vars (simulating WI manager setting them before crew run)
        token_file = tmp_path / "token.txt"
        token_file.write_text("eyJhbGciOiJSUzI1NiJ9.deferred")
        monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("AZURE_TENANT_ID", "deferred-tenant")
        monkeypatch.setenv("AZURE_CLIENT_ID", "deferred-client")

        mock_wi_cred = MagicMock()
        with patch(
            "azure.identity.WorkloadIdentityCredential",
            return_value=mock_wi_cred,
        ):
            kwargs = completion._make_client_kwargs()
            assert kwargs["credential"] is mock_wi_cred

    def test_make_client_kwargs_includes_api_version(self, monkeypatch):
        """api_version is included in client kwargs."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        monkeypatch.setenv("AZURE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)

        completion = AzureCompletion(model="gpt-4", api_version="2025-01-01")
        kwargs = completion._make_client_kwargs()
        assert kwargs["api_version"] == "2025-01-01"
        assert kwargs["endpoint"] == ENDPOINT
