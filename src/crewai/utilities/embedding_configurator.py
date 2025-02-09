import os
from typing import Any, Dict, List, Optional, cast

from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import validate_embedding_function

from crewai.utilities.exceptions.embedding_exceptions import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    EmbeddingInitializationError
)

# Rest of embedding_configurator.py content...
