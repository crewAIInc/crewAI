from __future__ import annotations

from functools import cached_property
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from crewai_tools.rag.misc import compute_sha256


if TYPE_CHECKING:
    from crewai_tools.rag.data_types import DataType


class SourceContent:
    def __init__(self, source: str | Path):
        self.source = str(source)

    def is_url(self) -> bool:
        if not isinstance(self.source, str):
            return False
        try:
            parsed_url = urlparse(self.source)
            return bool(parsed_url.scheme and parsed_url.netloc)
        except Exception:
            return False

    def path_exists(self) -> bool:
        return os.path.exists(self.source)

    @cached_property
    def data_type(self) -> DataType:
        from crewai_tools.rag.data_types import DataTypes

        return DataTypes.from_content(self.source)

    @cached_property
    def source_ref(self) -> str:
        """ "
        Returns the source reference for the content.
        If the content is a URL or a local file, returns the source.
        Otherwise, returns the hash of the content.
        """
        if self.is_url() or self.path_exists():
            return self.source

        return compute_sha256(self.source)
