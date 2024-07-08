import json
from datetime import datetime
from uuid import UUID

from openai import BaseModel


class CrewJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder for Crew related objects.
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)
