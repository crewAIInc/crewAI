import os
import re
from typing import Optional, Any, Union

from crewai.tools import BaseTool

class SerpApiBaseTool(BaseTool):
    """Base class for SerpApi functionality with shared capabilities."""

    client: Optional[Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            from serpapi import Client
        except ImportError:
            raise ImportError(
                "`serpapi` package not found, please install with `pip install serpapi`"
            )
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key, you can get the key from https://serpapi.com/manage-api-key"
            )
        self.client = Client(api_key=api_key)

    def _omit_fields(self, data: Union[dict, list], omit_patterns: list[str]) -> None:
        if isinstance(data, dict):
            for field in list(data.keys()):
                if any(re.compile(p).match(field) for p in omit_patterns):
                    data.pop(field, None)
                else:
                    if isinstance(data[field], (dict, list)):
                        self._omit_fields(data[field], omit_patterns)
        elif isinstance(data, list):
            for item in data:
                self._omit_fields(item, omit_patterns)
