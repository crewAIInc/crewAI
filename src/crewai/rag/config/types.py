from typing import TYPE_CHECKING, Annotated, TypeAlias
from pydantic import Field

from crewai.rag.config.constants import DISCRIMINATOR

if TYPE_CHECKING:
    from crewai.rag.chromadb.config import ChromaDBConfig
else:
    try:
        from crewai.rag.chromadb.config import ChromaDBConfig
    except ImportError:
        from crewai.rag.config.optional_imports.providers import (
            MissingChromaDBConfig as ChromaDBConfig,
        )

Providers: TypeAlias = ChromaDBConfig
RagConfigType: TypeAlias = Annotated[Providers, Field(discriminator=DISCRIMINATOR)]
