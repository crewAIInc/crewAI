from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Callable, Dict, Optional, Type, Tuple, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model, validator
from pydantic.fields import FieldInfo
from pydantic import BaseModel as PydanticBaseModel

def _create_model_fields(fields: Dict[str, Tuple[Any, FieldInfo]]) -> Dict[str, Any]:
    """Helper function to create model fields with proper type hints."""
    return {name: (annotation, field) for name, (annotation, field) in fields.items()}

# Rest of base_tool.py content...
