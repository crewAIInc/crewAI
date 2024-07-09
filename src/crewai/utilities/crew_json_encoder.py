from datetime import datetime
import json
from uuid import UUID
from pydantic import BaseModel


class CrewJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return self._handle_pydantic_model(obj)
        elif isinstance(obj, UUID):
            return str(obj)

        elif isinstance(obj, datetime):
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
