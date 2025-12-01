"""Type definitions for the embeddings module."""

from typing import Any, Literal, TypeAlias

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.aws.types import BedrockProviderSpec
from crewai.rag.embeddings.providers.cohere.types import CohereProviderSpec
from crewai.rag.embeddings.providers.custom.types import CustomProviderSpec
from crewai.rag.embeddings.providers.google.types import (
    GenerativeAiProviderSpec,
    VertexAIProviderSpec,
)
from crewai.rag.embeddings.providers.huggingface.types import HuggingFaceProviderSpec
from crewai.rag.embeddings.providers.ibm.types import (
    WatsonXProviderSpec,
)
from crewai.rag.embeddings.providers.instructor.types import InstructorProviderSpec
from crewai.rag.embeddings.providers.jina.types import JinaProviderSpec
from crewai.rag.embeddings.providers.microsoft.types import AzureProviderSpec
from crewai.rag.embeddings.providers.ollama.types import OllamaProviderSpec
from crewai.rag.embeddings.providers.onnx.types import ONNXProviderSpec
from crewai.rag.embeddings.providers.openai.types import OpenAIProviderSpec
from crewai.rag.embeddings.providers.openclip.types import OpenCLIPProviderSpec
from crewai.rag.embeddings.providers.roboflow.types import RoboflowProviderSpec
from crewai.rag.embeddings.providers.sentence_transformer.types import (
    SentenceTransformerProviderSpec,
)
from crewai.rag.embeddings.providers.text2vec.types import Text2VecProviderSpec
from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderSpec


ProviderSpec: TypeAlias = (
    AzureProviderSpec
    | BedrockProviderSpec
    | CohereProviderSpec
    | CustomProviderSpec
    | GenerativeAiProviderSpec
    | HuggingFaceProviderSpec
    | InstructorProviderSpec
    | JinaProviderSpec
    | OllamaProviderSpec
    | ONNXProviderSpec
    | OpenAIProviderSpec
    | OpenCLIPProviderSpec
    | RoboflowProviderSpec
    | SentenceTransformerProviderSpec
    | Text2VecProviderSpec
    | VertexAIProviderSpec
    | VoyageAIProviderSpec
    | WatsonXProviderSpec
)

AllowedEmbeddingProviders = Literal[
    "azure",
    "amazon-bedrock",
    "cohere",
    "custom",
    "google-generativeai",
    "google-vertex",
    "huggingface",
    "instructor",
    "jina",
    "ollama",
    "onnx",
    "openai",
    "openclip",
    "roboflow",
    "sentence-transformer",
    "text2vec",
    "voyageai",
    "watsonx",
]

EmbedderConfig: TypeAlias = (
    ProviderSpec | BaseEmbeddingsProvider[Any] | type[BaseEmbeddingsProvider[Any]]
)
