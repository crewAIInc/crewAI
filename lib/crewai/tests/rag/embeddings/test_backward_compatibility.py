"""Tests for backward compatibility of embedding provider configurations."""

from crewai.rag.embeddings.factory import build_embedder, PROVIDER_PATHS
from crewai.rag.embeddings.providers.openai.openai_provider import OpenAIProvider
from crewai.rag.embeddings.providers.cohere.cohere_provider import CohereProvider
from crewai.rag.embeddings.providers.google.generative_ai import GenerativeAiProvider
from crewai.rag.embeddings.providers.google.vertex import VertexAIProvider
from crewai.rag.embeddings.providers.microsoft.azure import AzureProvider
from crewai.rag.embeddings.providers.jina.jina_provider import JinaProvider
from crewai.rag.embeddings.providers.ollama.ollama_provider import OllamaProvider
from crewai.rag.embeddings.providers.aws.bedrock import BedrockProvider
from crewai.rag.embeddings.providers.text2vec.text2vec_provider import Text2VecProvider
from crewai.rag.embeddings.providers.sentence_transformer.sentence_transformer_provider import (
    SentenceTransformerProvider,
)
from crewai.rag.embeddings.providers.instructor.instructor_provider import InstructorProvider
from crewai.rag.embeddings.providers.openclip.openclip_provider import OpenCLIPProvider


class TestGoogleProviderAlias:
    """Test that 'google' provider name alias works for backward compatibility."""

    def test_google_alias_in_provider_paths(self):
        """Verify 'google' is registered as an alias for google-generativeai."""
        assert "google" in PROVIDER_PATHS
        assert "google-generativeai" in PROVIDER_PATHS
        assert PROVIDER_PATHS["google"] == PROVIDER_PATHS["google-generativeai"]


class TestModelKeyBackwardCompatibility:
    """Test that 'model' config key works as alias for 'model_name'."""

    def test_openai_provider_accepts_model_key(self):
        """Test OpenAI provider accepts 'model' as alias for 'model_name'."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="text-embedding-3-small",
        )
        assert provider.model_name == "text-embedding-3-small"

    def test_openai_provider_model_name_takes_precedence(self):
        """Test that model_name takes precedence when both are provided."""
        provider = OpenAIProvider(
            api_key="test-key",
            model_name="text-embedding-3-large",
        )
        assert provider.model_name == "text-embedding-3-large"

    def test_cohere_provider_accepts_model_key(self):
        """Test Cohere provider accepts 'model' as alias for 'model_name'."""
        provider = CohereProvider(
            api_key="test-key",
            model="embed-english-v3.0",
        )
        assert provider.model_name == "embed-english-v3.0"

    def test_google_generativeai_provider_accepts_model_key(self):
        """Test Google Generative AI provider accepts 'model' as alias."""
        provider = GenerativeAiProvider(
            api_key="test-key",
            model="gemini-embedding-001",
        )
        assert provider.model_name == "gemini-embedding-001"

    def test_google_vertex_provider_accepts_model_key(self):
        """Test Google Vertex AI provider accepts 'model' as alias."""
        provider = VertexAIProvider(
            api_key="test-key",
            model="text-embedding-004",
        )
        assert provider.model_name == "text-embedding-004"

    def test_azure_provider_accepts_model_key(self):
        """Test Azure provider accepts 'model' as alias for 'model_name'."""
        provider = AzureProvider(
            api_key="test-key",
            deployment_id="test-deployment",
            model="text-embedding-ada-002",
        )
        assert provider.model_name == "text-embedding-ada-002"

    def test_jina_provider_accepts_model_key(self):
        """Test Jina provider accepts 'model' as alias for 'model_name'."""
        provider = JinaProvider(
            api_key="test-key",
            model="jina-embeddings-v3",
        )
        assert provider.model_name == "jina-embeddings-v3"

    def test_ollama_provider_accepts_model_key(self):
        """Test Ollama provider accepts 'model' as alias for 'model_name'."""
        provider = OllamaProvider(
            model="nomic-embed-text",
        )
        assert provider.model_name == "nomic-embed-text"

    def test_text2vec_provider_accepts_model_key(self):
        """Test Text2Vec provider accepts 'model' as alias for 'model_name'."""
        provider = Text2VecProvider(
            model="shibing624/text2vec-base-multilingual",
        )
        assert provider.model_name == "shibing624/text2vec-base-multilingual"

    def test_sentence_transformer_provider_accepts_model_key(self):
        """Test SentenceTransformer provider accepts 'model' as alias."""
        provider = SentenceTransformerProvider(
            model="all-mpnet-base-v2",
        )
        assert provider.model_name == "all-mpnet-base-v2"

    def test_instructor_provider_accepts_model_key(self):
        """Test Instructor provider accepts 'model' as alias for 'model_name'."""
        provider = InstructorProvider(
            model="hkunlp/instructor-xl",
        )
        assert provider.model_name == "hkunlp/instructor-xl"

    def test_openclip_provider_accepts_model_key(self):
        """Test OpenCLIP provider accepts 'model' as alias for 'model_name'."""
        provider = OpenCLIPProvider(
            model="ViT-B-16",
        )
        assert provider.model_name == "ViT-B-16"


class TestTaskTypeConfiguration:
    """Test that task_type configuration works correctly."""

    def test_google_provider_accepts_lowercase_task_type(self):
        """Test Google provider accepts lowercase task_type."""
        provider = GenerativeAiProvider(
            api_key="test-key",
            task_type="retrieval_document",
        )
        assert provider.task_type == "retrieval_document"

    def test_google_provider_accepts_uppercase_task_type(self):
        """Test Google provider accepts uppercase task_type."""
        provider = GenerativeAiProvider(
            api_key="test-key",
            task_type="RETRIEVAL_QUERY",
        )
        assert provider.task_type == "RETRIEVAL_QUERY"

    def test_google_provider_default_task_type(self):
        """Test Google provider has correct default task_type."""
        provider = GenerativeAiProvider(
            api_key="test-key",
        )
        assert provider.task_type == "RETRIEVAL_DOCUMENT"


class TestFactoryBackwardCompatibility:
    """Test factory function with backward compatible configurations."""

    def test_factory_with_google_alias(self):
        """Test factory resolves 'google' to google-generativeai provider."""
        config = {
            "provider": "google",
            "config": {
                "api_key": "test-key",
                "model": "gemini-embedding-001",
            },
        }

        from unittest.mock import patch, MagicMock

        with patch("crewai.rag.embeddings.factory.import_and_validate_definition") as mock_import:
            mock_provider_class = MagicMock()
            mock_provider_instance = MagicMock()
            mock_import.return_value = mock_provider_class
            mock_provider_class.return_value = mock_provider_instance

            build_embedder(config)

            mock_import.assert_called_once_with(
                "crewai.rag.embeddings.providers.google.generative_ai.GenerativeAiProvider"
            )

    def test_factory_with_model_key_openai(self):
        """Test factory passes 'model' config to OpenAI provider."""
        config = {
            "provider": "openai",
            "config": {
                "api_key": "test-key",
                "model": "text-embedding-3-small",
            },
        }

        from unittest.mock import patch, MagicMock

        with patch("crewai.rag.embeddings.factory.import_and_validate_definition") as mock_import:
            mock_provider_class = MagicMock()
            mock_provider_instance = MagicMock()
            mock_import.return_value = mock_provider_class
            mock_provider_class.return_value = mock_provider_instance

            build_embedder(config)

            call_kwargs = mock_provider_class.call_args.kwargs
            assert call_kwargs["model"] == "text-embedding-3-small"


class TestDocumentationCodeSnippets:
    """Test code snippets from documentation work correctly."""

    def test_memory_openai_config(self):
        """Test OpenAI config from memory.mdx documentation."""
        provider = OpenAIProvider(
            model_name="text-embedding-3-small",
        )
        assert provider.model_name == "text-embedding-3-small"

    def test_memory_openai_config_with_options(self):
        """Test OpenAI config with all options from memory.mdx."""
        provider = OpenAIProvider(
            api_key="your-openai-api-key",
            model_name="text-embedding-3-large",
            dimensions=1536,
            organization_id="your-org-id",
        )
        assert provider.model_name == "text-embedding-3-large"
        assert provider.dimensions == 1536

    def test_memory_azure_config(self):
        """Test Azure config from memory.mdx documentation."""
        provider = AzureProvider(
            api_key="your-azure-key",
            api_base="https://your-resource.openai.azure.com/",
            api_type="azure",
            api_version="2023-05-15",
            model_name="text-embedding-3-small",
            deployment_id="your-deployment-name",
        )
        assert provider.model_name == "text-embedding-3-small"
        assert provider.api_type == "azure"

    def test_memory_google_generativeai_config(self):
        """Test Google Generative AI config from memory.mdx documentation."""
        provider = GenerativeAiProvider(
            api_key="your-google-api-key",
            model_name="gemini-embedding-001",
        )
        assert provider.model_name == "gemini-embedding-001"

    def test_memory_cohere_config(self):
        """Test Cohere config from memory.mdx documentation."""
        provider = CohereProvider(
            api_key="your-cohere-api-key",
            model_name="embed-english-v3.0",
        )
        assert provider.model_name == "embed-english-v3.0"

    def test_knowledge_agent_embedder_config(self):
        """Test agent embedder config from knowledge.mdx documentation."""
        provider = GenerativeAiProvider(
            model_name="gemini-embedding-001",
            api_key="your-google-key",
        )
        assert provider.model_name == "gemini-embedding-001"

    def test_ragtool_openai_config(self):
        """Test RagTool OpenAI config from ragtool.mdx documentation."""
        provider = OpenAIProvider(
            model_name="text-embedding-3-small",
        )
        assert provider.model_name == "text-embedding-3-small"

    def test_ragtool_cohere_config(self):
        """Test RagTool Cohere config from ragtool.mdx documentation."""
        provider = CohereProvider(
            api_key="your-api-key",
            model_name="embed-english-v3.0",
        )
        assert provider.model_name == "embed-english-v3.0"

    def test_ragtool_ollama_config(self):
        """Test RagTool Ollama config from ragtool.mdx documentation."""
        provider = OllamaProvider(
            model_name="llama2",
            url="http://localhost:11434/api/embeddings",
        )
        assert provider.model_name == "llama2"

    def test_ragtool_azure_config(self):
        """Test RagTool Azure config from ragtool.mdx documentation."""
        provider = AzureProvider(
            deployment_id="your-deployment-id",
            api_key="your-api-key",
            api_base="https://your-resource.openai.azure.com",
            api_version="2024-02-01",
            model_name="text-embedding-ada-002",
            api_type="azure",
        )
        assert provider.model_name == "text-embedding-ada-002"
        assert provider.deployment_id == "your-deployment-id"

    def test_ragtool_google_generativeai_config(self):
        """Test RagTool Google Generative AI config from ragtool.mdx."""
        provider = GenerativeAiProvider(
            api_key="your-api-key",
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
        )
        assert provider.model_name == "gemini-embedding-001"
        assert provider.task_type == "RETRIEVAL_DOCUMENT"

    def test_ragtool_jina_config(self):
        """Test RagTool Jina config from ragtool.mdx documentation."""
        provider = JinaProvider(
            api_key="your-api-key",
            model_name="jina-embeddings-v3",
        )
        assert provider.model_name == "jina-embeddings-v3"

    def test_ragtool_sentence_transformer_config(self):
        """Test RagTool SentenceTransformer config from ragtool.mdx."""
        provider = SentenceTransformerProvider(
            model_name="all-mpnet-base-v2",
            device="cuda",
            normalize_embeddings=True,
        )
        assert provider.model_name == "all-mpnet-base-v2"
        assert provider.device == "cuda"
        assert provider.normalize_embeddings is True


class TestLegacyConfigurationFormats:
    """Test legacy configuration formats that should still work."""

    def test_legacy_google_with_model_key(self):
        """Test legacy Google config using 'model' instead of 'model_name'."""
        provider = GenerativeAiProvider(
            api_key="test-key",
            model="text-embedding-005",
            task_type="retrieval_document",
        )
        assert provider.model_name == "text-embedding-005"
        assert provider.task_type == "retrieval_document"

    def test_legacy_openai_with_model_key(self):
        """Test legacy OpenAI config using 'model' instead of 'model_name'."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="text-embedding-ada-002",
        )
        assert provider.model_name == "text-embedding-ada-002"

    def test_legacy_cohere_with_model_key(self):
        """Test legacy Cohere config using 'model' instead of 'model_name'."""
        provider = CohereProvider(
            api_key="test-key",
            model="embed-multilingual-v3.0",
        )
        assert provider.model_name == "embed-multilingual-v3.0"

    def test_legacy_azure_with_model_key(self):
        """Test legacy Azure config using 'model' instead of 'model_name'."""
        provider = AzureProvider(
            api_key="test-key",
            deployment_id="test-deployment",
            model="text-embedding-3-large",
        )
        assert provider.model_name == "text-embedding-3-large"