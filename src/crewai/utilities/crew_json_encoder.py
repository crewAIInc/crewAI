"""JSON encoder for handling CrewAI specific types."""

import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class CrewJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for CrewAI objects and special types."""

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return self._handle_pydantic_model(obj)
        if isinstance(obj, (UUID, Decimal, Enum)):
            return str(obj)

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        return super().default(obj)

    def _handle_pydantic_model(self, obj):
        try:
            data = obj.model_dump()
            # Remove circular references
            for key, value in data.items():
                if isinstance(value, BaseModel):
                    data[key] = str(
                        value,
                    )  # Convert nested models to string representation
            return data
        except RecursionError:
            return str(
                obj,
            )  # Fall back to string representation if circular reference is detected
