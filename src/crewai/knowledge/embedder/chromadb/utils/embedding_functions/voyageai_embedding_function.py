import logging
from typing import List, Optional

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, api_key: str, model_name: str = "voyage-3"):
        try:
            import voyageai
        except ImportError:
            raise ValueError(
                "The voyageai python package is not installed. Please install it with `pip install voyageai`"
            )

        self._api_key = api_key
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        try:
            import voyageai
        except ImportError:
            raise ValueError(
                "The voyageai python package is not installed. Please install it with `pip install voyageai`"
            )

        if not input:
            return []

        if isinstance(input, str):
            input = [input]

        try:
            embeddings = voyageai.get_embeddings(
                input, model=self._model_name, api_key=self._api_key
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error during VoyageAI embedding: {e}")
            raise e
