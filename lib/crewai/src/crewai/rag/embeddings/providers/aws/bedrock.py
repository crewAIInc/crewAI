"""Amazon Bedrock embeddings provider."""

from typing import Any

from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import (
    AmazonBedrockEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


def create_aws_session() -> Any:
    """Create an AWS session for Bedrock.

    Returns:
        boto3.Session: AWS session object

    Raises:
        ImportError: If boto3 is not installed
        ValueError: If AWS session creation fails
    """
    try:
        import boto3

        return boto3.Session()
    except ImportError as e:
        raise ImportError(
            "boto3 is required for amazon-bedrock embeddings. "
            "Install it with: uv add boto3"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to create AWS session for amazon-bedrock. "
            f"Ensure AWS credentials are configured. Error: {e}"
        ) from e


class BedrockProvider(BaseEmbeddingsProvider[AmazonBedrockEmbeddingFunction]):
    """Amazon Bedrock embeddings provider."""

    embedding_callable: type[AmazonBedrockEmbeddingFunction] = Field(
        default=AmazonBedrockEmbeddingFunction,
        description="Amazon Bedrock embedding function class",
    )
    model_name: str = Field(
        default="amazon.titan-embed-text-v1",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_BEDROCK_MODEL_NAME",
            "BEDROCK_MODEL_NAME",
            "AWS_BEDROCK_MODEL_NAME",
            "model",
        ),
    )
    session: Any = Field(
        default_factory=create_aws_session, description="AWS session object"
    )
