"""
Centralized factory for creating type-safe Pydantic models.

Eliminates scattered `create_model` calls throughout the codebase
and provides consistent type handling with proper mypy compliance.
"""

from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, create_model


def create_tool_schema(
    model_name: str, 
    field_definitions: Dict[str, Any],
    base_class: Optional[Type[BaseModel]] = None
) -> Type[BaseModel]:
    """
    Create a Pydantic model for tool schema with proper type safety.
    
    Args:
        model_name: Name for the generated model class
        field_definitions: Dictionary mapping field names to type definitions
        base_class: Optional base class to inherit from
        
    Returns:
        Generated Pydantic model class
    """
    if base_class:
        return create_model(model_name, __base__=base_class, **field_definitions)  # type: ignore[call-overload]
    else:
        return create_model(model_name, **field_definitions)  # type: ignore[call-overload]