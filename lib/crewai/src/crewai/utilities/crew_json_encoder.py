"""JSON encoder for handling CrewAI specific types."""

import base64
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
import json
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from pydantic_core import PydanticSerializationError, to_jsonable_python


class CrewJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for CrewAI objects and special types."""

    def default(self, obj: Any) -> Any:
        """Custom serialization for CrewAI specific types.

        Args:
            obj: The object to serialize.

        Returns:
            A JSON-serializable representation of the object.
        """
        if isinstance(obj, BaseModel):
            return self._handle_pydantic_model(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (UUID, Decimal)):
            return str(obj)

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")

        try:
            return to_jsonable_python(obj)
        except PydanticSerializationError:
            return super().default(obj)

    @staticmethod
    def _handle_pydantic_model(obj: BaseModel) -> str | Any:
        """Preserve model dump shapes, including non-dictionary root values."""
        try:
            data = obj.model_dump()
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, BaseModel):
                        data[key] = str(value)
            return data
        except RecursionError:
            return str(obj)
