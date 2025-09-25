"""Amazon Bedrock embeddings provider."""

from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import (
    AmazonBedrockEmbeddingFunction,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider

try:
    from boto3.session import Session  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "boto3 is required for amazon-bedrock embeddings. Install it with: uv add boto3"
    ) from exc


def create_aws_session() -> Session:
    """Create an AWS session for Bedrock.

    Returns:
        boto3.Session: AWS session object

    Raises:
        ImportError: If boto3 is not installed
        ValueError: If AWS session creation fails
    """
    try:
        import boto3  # type: ignore[import]

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
        validation_alias="BEDROCK_MODEL_NAME",
    )
    session: Session = Field(
        default_factory=create_aws_session, description="AWS session object"
    )
