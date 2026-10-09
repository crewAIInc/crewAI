"""Test DeepInfra embedder configuration with factory."""

from unittest.mock import MagicMock, patch

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
import httpx
import pytest
from pydantic import ValidationError

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.providers.deepinfra import deepinfra_provider
from crewai.rag.embeddings.providers.deepinfra.deepinfra_provider import (
    DEFAULT_MODEL,
    DeepInfraProvider,
    effective_api_base,
)


PROVIDER_MODULE = "crewai.rag.embeddings.providers.deepinfra.deepinfra_provider"
CATALOG_URL = "https://api.deepinfra.com/v1/openai/models"
CATALOG_PARAMS = {"filter": "with_meta", "sort_by": "crewai"}


def _catalog_response(*entries: tuple[str, list[str]]) -> MagicMock:
    """Build a fake models-list response from (id, tags) pairs in catalog order."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "metadata": {"tags": tags}}
            for model_id, tags in entries
        ],
    }
    return response


@pytest.fixture(autouse=True)
def catalog_get(monkeypatch):
    """Keep the catalog lookup off the network, uncached and env-free between tests."""
    monkeypatch.delenv("DEEPINFRA_API_BASE", raising=False)
    deepinfra_provider._catalog_picks.clear()
    with patch(f"{PROVIDER_MODULE}.httpx.get", side_effect=httpx.ConnectError("offline")) as mock_get:
        yield mock_get
    deepinfra_provider._catalog_picks.clear()


class TestDeepInfraEmbedderFactory:
    """Test DeepInfra embedder configuration with factory function."""

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_deepinfra_with_nested_config(self, mock_import):
        """Test DeepInfra configuration with nested config key."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
            "provider": "deepinfra",
            "config": {
                "api_key": "test-deepinfra-key",
                "model_name": "Qwen/Qwen3-Embedding-4B",
                "api_base": "https://api.deepinfra.com/v1/openai",
                "default_headers": {"X-Test": "crewai"},
            },
        }

        result = build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.deepinfra.deepinfra_provider.DeepInfraProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-deepinfra-key"
        assert call_kwargs["model_name"] == "Qwen/Qwen3-Embedding-4B"
        assert call_kwargs["api_base"] == "https://api.deepinfra.com/v1/openai"
        assert call_kwargs["default_headers"] == {"X-Test": "crewai"}

        assert result == mock_embedding_function

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_deepinfra_with_model_alias(self, mock_import):
        """Test DeepInfra configuration with 'model' alias instead of 'model_name'."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
            "provider": "deepinfra",
            "config": {
                "api_key": "test-deepinfra-key",
                "model": "intfloat/multilingual-e5-large",
            },
        }

        result = build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.deepinfra.deepinfra_provider.DeepInfraProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-deepinfra-key"
        assert call_kwargs["model"] == "intfloat/multilingual-e5-large"

        assert result == mock_embedding_function

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_deepinfra_import_error(self, mock_import):
        """Test handling of import errors for DeepInfra provider."""
        mock_import.side_effect = ImportError("Failed to import DeepInfra provider")

        embedder_config = {
            "provider": "deepinfra",
            "config": {"api_key": "test-key"},
        }

        with pytest.raises(ImportError) as exc_info:
            build_embedder(embedder_config)

        assert "Failed to import provider deepinfra" in str(exc_info.value)

    def test_model_alias_reaches_embedding_function_as_model_name(self):
        """Test the 'model' alias is forwarded as model_name, not as a stray kwarg."""
        embedder = build_embedder(
            {
                "provider": "deepinfra",
                "config": {"api_key": "test-key", "model": "Qwen/Qwen3-Embedding-4B"},
            }
        )

        assert isinstance(embedder, OpenAIEmbeddingFunction)
        assert embedder.model_name == "Qwen/Qwen3-Embedding-4B"

    def test_build_embedder_without_model_uses_catalog_model(self, catalog_get):
        """Test an embedder built without a model embeds with the catalog's pick."""
        catalog_get.side_effect = None
        catalog_get.return_value = _catalog_response(
            ("deepseek-ai/DeepSeek-V4-Flash", ["chat"]),
            ("BAAI/bge-m3", ["embed"]),
        )

        embedder = build_embedder(
            {"provider": "deepinfra", "config": {"api_key": "test-key"}}
        )

        assert isinstance(embedder, OpenAIEmbeddingFunction)
        assert embedder.model_name == "BAAI/bge-m3"


class TestDeepInfraDefaultModel:
    """Test how the default model is resolved from the DeepInfra catalog."""

    def test_first_embedding_model_in_catalog_order(self, catalog_get):
        """Test the first model tagged 'embed' wins, in the order the catalog sends."""
        catalog_get.side_effect = None
        catalog_get.return_value = _catalog_response(
            ("deepseek-ai/DeepSeek-V4-Flash", ["chat", "reasoning"]),
            ("Qwen/Qwen3-Embedding-4B", ["embed"]),
            ("BAAI/bge-m3", ["embed"]),
        )

        provider = DeepInfraProvider(api_key="test-key")

        assert provider.model_name == "Qwen/Qwen3-Embedding-4B"
        catalog_get.assert_called_once_with(
            CATALOG_URL, params=CATALOG_PARAMS, timeout=3.0
        )

    def test_custom_api_base_is_queried(self, catalog_get):
        """Test the catalog is read from the configured api_base."""
        catalog_get.side_effect = None
        catalog_get.return_value = _catalog_response(("BAAI/bge-m3", ["embed"]))

        provider = DeepInfraProvider(
            api_key="test-key", api_base="https://proxy.example.com/v1/openai/"
        )

        assert provider.model_name == "BAAI/bge-m3"
        assert catalog_get.call_args.args[0] == "https://proxy.example.com/v1/openai/models"

    @pytest.mark.parametrize("key", ["model_name", "model"])
    def test_explicit_model_skips_catalog(self, catalog_get, key):
        """Test a configured model is used as-is without reading the catalog."""
        provider = DeepInfraProvider(api_key="test-key", **{key: "BAAI/bge-m3"})

        assert provider.model_name == "BAAI/bge-m3"
        catalog_get.assert_not_called()

    def test_falls_back_when_catalog_unreachable(self, catalog_get):
        """Test a connection error leaves the static default in place."""
        provider = DeepInfraProvider(api_key="test-key")

        assert provider.model_name == DEFAULT_MODEL == "Qwen/Qwen3-Embedding-8B"
        catalog_get.assert_called_once()

    def test_falls_back_on_http_error(self, catalog_get):
        """Test an HTTP error status leaves the static default in place."""
        response = MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=MagicMock()
        )
        catalog_get.side_effect = None
        catalog_get.return_value = response

        assert DeepInfraProvider(api_key="test-key").model_name == DEFAULT_MODEL

    @pytest.mark.parametrize(
        "payload",
        [{"object": "list"}, {"data": "nope"}, [{"id": "x"}]],
        ids=["no-data-key", "data-not-a-list", "not-an-object"],
    )
    def test_falls_back_on_malformed_payload(self, catalog_get, payload):
        """Test an unexpected payload shape leaves the static default in place."""
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        catalog_get.side_effect = None
        catalog_get.return_value = response

        assert DeepInfraProvider(api_key="test-key").model_name == DEFAULT_MODEL

    def test_falls_back_on_invalid_json(self, catalog_get):
        """Test a non-JSON body leaves the static default in place."""
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("not json")
        catalog_get.side_effect = None
        catalog_get.return_value = response

        assert DeepInfraProvider(api_key="test-key").model_name == DEFAULT_MODEL

    def test_falls_back_when_no_embedding_model_tagged(self, catalog_get):
        """Test a catalog without 'embed' tags leaves the static default in place."""
        catalog_get.side_effect = None
        catalog_get.return_value = _catalog_response(
            ("deepseek-ai/DeepSeek-V4-Flash", ["chat"]),
            ("untagged/model", []),
        )

        assert DeepInfraProvider(api_key="test-key").model_name == DEFAULT_MODEL

    def test_successful_pick_is_cached_per_api_base(self, catalog_get):
        """Test the catalog is read once per base URL, not once per provider."""
        catalog_get.side_effect = None
        catalog_get.return_value = _catalog_response(("BAAI/bge-m3", ["embed"]))

        DeepInfraProvider(api_key="test-key")
        DeepInfraProvider(api_key="test-key")
        DeepInfraProvider(api_key="test-key", api_base="https://proxy.example.com/v1")

        assert catalog_get.call_count == 2

    def test_failed_lookup_is_not_cached(self, catalog_get):
        """Test a failed lookup does not pin the fallback for later providers."""
        assert DeepInfraProvider(api_key="test-key").model_name == DEFAULT_MODEL

        catalog_get.side_effect = None
        catalog_get.return_value = _catalog_response(("BAAI/bge-m3", ["embed"]))

        assert DeepInfraProvider(api_key="test-key").model_name == "BAAI/bge-m3"
        assert catalog_get.call_count == 2

    def test_effective_api_base_precedence(self, monkeypatch):
        """Test an explicit base beats DEEPINFRA_API_BASE, which beats the default."""
        assert effective_api_base() == "https://api.deepinfra.com/v1/openai"

        monkeypatch.setenv("DEEPINFRA_API_BASE", "https://env.example.com/v1/openai")

        assert effective_api_base() == "https://env.example.com/v1/openai"
        assert effective_api_base("https://explicit.example.com/v1") == "https://explicit.example.com/v1"


class TestDeepInfraProviderDirect:
    """Test DeepInfraProvider Pydantic settings model directly."""

    def test_default_values(self):
        """Test default values for DeepInfraProvider."""
        provider = DeepInfraProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model_name == "Qwen/Qwen3-Embedding-8B"
        assert provider.api_base == "https://api.deepinfra.com/v1/openai"
        assert provider.default_headers is None

    def test_custom_values(self):
        """Test custom configuration values."""
        provider = DeepInfraProvider(
            api_key="test-custom-key",
            model="Qwen/Qwen3-Embedding-4B",
            api_base="https://proxy.example.com/v1/openai",
            default_headers={"X-Test": "crewai"},
        )

        assert provider.api_key == "test-custom-key"
        assert provider.model_name == "Qwen/Qwen3-Embedding-4B"
        assert provider.api_base == "https://proxy.example.com/v1/openai"
        assert provider.default_headers == {"X-Test": "crewai"}

    def test_dimensions_is_rejected(self):
        """Test dimensions fails loudly instead of being dropped by the embedding function."""
        with pytest.raises(ValidationError, match="dimensions is not supported"):
            DeepInfraProvider(api_key="test-key", dimensions=1024)

    def test_dimensions_is_rejected_through_factory(self):
        """Test an embedder config with dimensions fails at build time with the reason."""
        with pytest.raises(ValidationError, match="text-embedding-3"):
            build_embedder(
                {
                    "provider": "deepinfra",
                    "config": {"api_key": "test-key", "dimensions": 1024},
                }
            )

    def test_dimensions_none_is_accepted(self):
        """Test an explicit None for dimensions is a no-op."""
        provider = DeepInfraProvider(api_key="test-key", dimensions=None)

        assert provider.model_name == "Qwen/Qwen3-Embedding-8B"

    def test_missing_api_key_raises_validation_error(self, monkeypatch):
        """Test that missing API key raises ValidationError when no env vars set."""
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)

        with pytest.raises(ValidationError):
            DeepInfraProvider()

    def test_env_var_deepinfra_api_key(self, monkeypatch):
        """Test resolving API key from DEEPINFRA_API_KEY env var."""
        monkeypatch.setenv("DEEPINFRA_API_KEY", "di-env-key")

        provider = DeepInfraProvider()
        assert provider.api_key == "di-env-key"

    def test_env_var_deepinfra_api_base(self, monkeypatch, catalog_get):
        """Test resolving the base URL from DEEPINFRA_API_BASE env var."""
        monkeypatch.setenv("DEEPINFRA_API_KEY", "di-env-key")
        monkeypatch.setenv("DEEPINFRA_API_BASE", "https://proxy.example.com/v1/openai")

        provider = DeepInfraProvider()

        assert provider.api_base == "https://proxy.example.com/v1/openai"
        assert catalog_get.call_args.args[0] == "https://proxy.example.com/v1/openai/models"

    def test_model_alias_normalization(self):
        """Test 'model' parameter maps to 'model_name'."""
        provider = DeepInfraProvider(
            api_key="test-key",
            model="intfloat/multilingual-e5-large",
        )
        assert provider.model_name == "intfloat/multilingual-e5-large"

    def test_model_name_takes_precedence(self):
        """Test that model_name takes precedence over model if both are given."""
        provider = DeepInfraProvider(
            api_key="test-key",
            model="BAAI/bge-m3",
            model_name="Qwen/Qwen3-Embedding-4B",
        )
        assert provider.model_name == "Qwen/Qwen3-Embedding-4B"
