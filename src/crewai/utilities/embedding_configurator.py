import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, cast

# Initialize with None to indicate module import status
CHROMADB_AVAILABLE = False

# Define placeholder types for when chromadb is not available
class EmbeddingFunction:
    def __call__(self, texts):
        raise NotImplementedError("Chromadb is not available")

Documents = List[str]
Embeddings = List[List[float]]

def validate_embedding_function(func):
    return func

# Try to import chromadb-related modules with proper error handling
try:
    from chromadb.api.types import Documents as ChromaDocuments
    from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction
    from chromadb.api.types import Embeddings as ChromaEmbeddings
    from chromadb.utils import (
        validate_embedding_function as chroma_validate_embedding_function,
    )
    
    # Override our placeholder types with the real ones
    Documents = ChromaDocuments
    EmbeddingFunction = ChromaEmbeddingFunction
    Embeddings = ChromaEmbeddings
    validate_embedding_function = chroma_validate_embedding_function
    
    CHROMADB_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # This captures both ImportError and AttributeError (which can happen with NumPy 2.x)
    warnings.warn(f"Failed to import chromadb: {str(e)}. Embedding functionality will be limited.")


class EmbeddingConfigurator:
    def __init__(self):
        self.embedding_functions = {
            "openai": self._configure_openai,
            "azure": self._configure_azure,
            "ollama": self._configure_ollama,
            "vertexai": self._configure_vertexai,
            "google": self._configure_google,
            "cohere": self._configure_cohere,
            "voyageai": self._configure_voyageai,
            "bedrock": self._configure_bedrock,
            "huggingface": self._configure_huggingface,
            "watson": self._configure_watson,
            "custom": self._configure_custom,
        }

    def configure_embedder(
        self,
        embedder_config: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingFunction:
        """Configures and returns an embedding function based on the provided config."""
        if not CHROMADB_AVAILABLE:
            return self._create_unavailable_embedding_function()
            
        if embedder_config is None:
            return self._create_default_embedding_function()

        provider = embedder_config.get("provider")
        config = embedder_config.get("config", {})
        model_name = config.get("model") if provider != "custom" else None

        if provider not in self.embedding_functions:
            raise Exception(
                f"Unsupported embedding provider: {provider}, supported providers: {list(self.embedding_functions.keys())}"
            )

        embedding_function = self.embedding_functions[provider]
        return (
            embedding_function(config)
            if provider == "custom"
            else embedding_function(config, model_name)
        )
        
    @staticmethod
    def _create_unavailable_embedding_function():
        """Creates a fallback embedding function when chromadb is not available."""
        class UnavailableEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input):
                raise ImportError(
                    "Chromadb is not available due to NumPy compatibility issues. "
                    "Either downgrade to NumPy<2 or upgrade chromadb and related dependencies."
                )
        
        return UnavailableEmbeddingFunction()

    @staticmethod
    def _create_default_embedding_function():
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.openai_embedding_function import (
                OpenAIEmbeddingFunction,
            )

            return OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
            )
        except (ImportError, AttributeError) as e:
            import warnings
            warnings.warn(f"Failed to import OpenAIEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_openai(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.openai_embedding_function import (
                OpenAIEmbeddingFunction,
            )

            return OpenAIEmbeddingFunction(
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                model_name=model_name,
                api_base=config.get("api_base", None),
                api_type=config.get("api_type", None),
                api_version=config.get("api_version", None),
                default_headers=config.get("default_headers", None),
                dimensions=config.get("dimensions", None),
                deployment_id=config.get("deployment_id", None),
                organization_id=config.get("organization_id", None),
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import OpenAIEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_azure(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.openai_embedding_function import (
                OpenAIEmbeddingFunction,
            )

            return OpenAIEmbeddingFunction(
                api_key=config.get("api_key"),
                api_base=config.get("api_base"),
                api_type=config.get("api_type", "azure"),
                api_version=config.get("api_version"),
                model_name=model_name,
                default_headers=config.get("default_headers"),
                dimensions=config.get("dimensions"),
                deployment_id=config.get("deployment_id"),
                organization_id=config.get("organization_id"),
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import OpenAIEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_ollama(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.ollama_embedding_function import (
                OllamaEmbeddingFunction,
            )

            return OllamaEmbeddingFunction(
                url=config.get("url", "http://localhost:11434/api/embeddings"),
                model_name=model_name,
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import OllamaEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_vertexai(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.google_embedding_function import (
                GoogleVertexEmbeddingFunction,
            )

            return GoogleVertexEmbeddingFunction(
                model_name=model_name,
                api_key=config.get("api_key"),
                project_id=config.get("project_id"),
                region=config.get("region"),
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import GoogleVertexEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_google(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.google_embedding_function import (
                GoogleGenerativeAiEmbeddingFunction,
            )

            return GoogleGenerativeAiEmbeddingFunction(
                model_name=model_name,
                api_key=config.get("api_key"),
                task_type=config.get("task_type"),
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import GoogleGenerativeAiEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_cohere(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.cohere_embedding_function import (
                CohereEmbeddingFunction,
            )

            return CohereEmbeddingFunction(
                model_name=model_name,
                api_key=config.get("api_key"),
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import CohereEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_voyageai(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.voyageai_embedding_function import (
                VoyageAIEmbeddingFunction,
            )

            return VoyageAIEmbeddingFunction(
                model_name=model_name,
                api_key=config.get("api_key"),
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import VoyageAIEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_bedrock(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import (
                AmazonBedrockEmbeddingFunction,
            )

            # Allow custom model_name override with backwards compatibility
            kwargs = {"session": config.get("session")}
            if model_name is not None:
                kwargs["model_name"] = model_name
            return AmazonBedrockEmbeddingFunction(**kwargs)
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import AmazonBedrockEmbeddingFunction: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_huggingface(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            from chromadb.utils.embedding_functions.huggingface_embedding_function import (
                HuggingFaceEmbeddingServer,
            )

            return HuggingFaceEmbeddingServer(
                url=config.get("api_url"),
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to import HuggingFaceEmbeddingServer: {str(e)}")
            return EmbeddingConfigurator._create_unavailable_embedding_function()

    @staticmethod
    def _configure_watson(config, model_name):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        try:
            import ibm_watsonx_ai.foundation_models as watson_models
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
        except ImportError as e:
            warnings.warn(
                "IBM Watson dependencies are not installed. Please install them to use Watson embedding."
            )
            return EmbeddingConfigurator._create_unavailable_embedding_function()

        class WatsonEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]

                embed_params = {
                    EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
                    EmbedParams.RETURN_OPTIONS: {"input_text": True},
                }

                embedding = watson_models.Embeddings(
                    model_id=config.get("model"),
                    params=embed_params,
                    credentials=Credentials(
                        api_key=config.get("api_key"), url=config.get("api_url")
                    ),
                    project_id=config.get("project_id"),
                )

                try:
                    embeddings = embedding.embed_documents(input)
                    return cast(Embeddings, embeddings)
                except Exception as e:
                    print("Error during Watson embedding:", e)
                    raise e

        return WatsonEmbeddingFunction()

    @staticmethod
    def _configure_custom(config):
        if not CHROMADB_AVAILABLE:
            return EmbeddingConfigurator._create_unavailable_embedding_function()
            
        custom_embedder = config.get("embedder")
        if isinstance(custom_embedder, EmbeddingFunction):
            try:
                validate_embedding_function(custom_embedder)
                return custom_embedder
            except Exception as e:
                warnings.warn(f"Invalid custom embedding function: {str(e)}")
                return EmbeddingConfigurator._create_unavailable_embedding_function()
        elif callable(custom_embedder):
            try:
                instance = custom_embedder()
                if isinstance(instance, EmbeddingFunction):
                    validate_embedding_function(instance)
                    return instance
                warnings.warn("Custom embedder does not create an EmbeddingFunction instance")
                return EmbeddingConfigurator._create_unavailable_embedding_function()
            except Exception as e:
                warnings.warn(f"Error instantiating custom embedder: {str(e)}")
                return EmbeddingConfigurator._create_unavailable_embedding_function()
        else:
            warnings.warn(
                "Custom embedder must be an instance of `EmbeddingFunction` or a callable that creates one"
            )
            return EmbeddingConfigurator._create_unavailable_embedding_function()
