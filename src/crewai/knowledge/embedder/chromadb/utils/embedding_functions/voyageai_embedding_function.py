import logging
from typing import List, Union

from chromadb.api.types import Documents, EmbeddingFunction

logger = logging.getLogger(__name__)


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    VoyageAI embedding function for ChromaDB.
    
    This class provides integration with VoyageAI's embedding models for use with ChromaDB.
    It supports various VoyageAI models including voyage-3, voyage-3.5, and voyage-3.5-lite.
    
    Attributes:
        _api_key (str): The API key for VoyageAI.
        _model_name (str): The name of the VoyageAI model to use.
    """

    def __init__(self, api_key: str, model_name: str = "voyage-3"):
        """
        Initialize the VoyageAI embedding function.
        
        Args:
            api_key (str): The API key for VoyageAI.
            model_name (str, optional): The name of the VoyageAI model to use. 
                Defaults to "voyage-3".
        
        Raises:
            ValueError: If the voyageai package is not installed or if the API key is empty.
        """
        self._ensure_voyageai_installed()
        
        if not api_key:
            raise ValueError("API key is required for VoyageAI embeddings")
        
        self._api_key = api_key
        self._model_name = model_name

    def _ensure_voyageai_installed(self):
        """
        Ensure that the voyageai package is installed.
        
        Raises:
            ValueError: If the voyageai package is not installed.
        """
        try:
            import voyageai  # noqa: F401
        except ImportError:
            raise ValueError(
                "The voyageai python package is not installed. Please install it with `pip install voyageai`"
            )

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for the input text(s).
        
        Args:
            input (Union[str, List[str]]): The text or list of texts to generate embeddings for.
        
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        
        Raises:
            ValueError: If the input is not a string or list of strings.
            voyageai.VoyageError: If there is an error with the VoyageAI API.
        """
        self._ensure_voyageai_installed()
        import voyageai

        if not input:
            return []

        if not isinstance(input, (str, list)):
            raise ValueError("Input must be a string or a list of strings")

        if isinstance(input, str):
            input = [input]

        try:
            embeddings = voyageai.get_embeddings(
                input, model=self._model_name, api_key=self._api_key
            )
            return embeddings
        except voyageai.VoyageError as e:
            logger.error(f"VoyageAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during VoyageAI embedding: {e}")
            raise
