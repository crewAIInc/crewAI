import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class CrewJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return self._handle_pydantic_model(obj)
        elif isinstance(obj, UUID) or isinstance(obj, Decimal) or isinstance(obj, Enum):
            return str(obj)

        elif isinstance(obj, datetime) or isinstance(obj, date):
            return obj.isoformat()

        return super().default(obj)

    def _handle_pydantic_model(self, obj):
        try:
            data = obj.model_dump()
            # Remove circular references
            for key, value in data.items():
                if isinstance(value, BaseModel):
                    data[key] = str(
                        value
                    )  # Convert nested models to string representation
            return data
        except RecursionError:
            return str(
                obj
            )  # Fall back to string representation if circular reference is detected
