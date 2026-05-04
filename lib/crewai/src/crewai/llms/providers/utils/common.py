import logging
import re
from typing import Any

from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.string_utils import sanitize_tool_name


def validate_function_name(name: str, provider: str = "LLM") -> str:
    """Validate function name according to common LLM provider requirements.

    Args:
        name: The function name to validate
        provider: The provider name for error messages

    Returns:
        The validated function name (unchanged if valid)

    Raises:
        ValueError: If the function name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"{provider} function name cannot be empty")

    if not (name[0].isalpha() or name[0] == "_"):
        raise ValueError(
            f"{provider} function name '{name}' must start with a letter or underscore"
        )

    if len(name) > 64:
        raise ValueError(
            f"{provider} function name '{name}' exceeds 64 character limit"
        )

    if not re.match(r"^[a-z_][a-z0-9_]*$", name):
        raise ValueError(
            f"{provider} function name '{name}' contains invalid characters. "
            f"Only lowercase letters, numbers, and underscores allowed"
        )

    return name


def extract_tool_info(tool: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Extract tool information from various schema formats.

    Handles both OpenAI/standard format and direct format:
    - OpenAI format: {"type": "function", "function": {"name": "...", ...}}
    - Direct format: {"name": "...", "description": "...", ...}

    Args:
        tool: Tool dictionary in any supported format

    Returns:
        Tuple of (name, description, parameters)

    Raises:
        ValueError: If tool format is invalid
    """
    if not isinstance(tool, dict):
        raise ValueError("Tool must be a dictionary")

    # Handle nested function schema format (OpenAI/standard)
    if "function" in tool:
        function_info = tool["function"]
        if not isinstance(function_info, dict):
            raise ValueError("Tool function must be a dictionary")

        name = function_info.get("name", "")
        description = function_info.get("description", "")
        parameters = function_info.get("parameters", {})
    else:
        # Direct format
        name = tool.get("name", "")
        description = tool.get("description", "")
        parameters = tool.get("parameters", {})

        # Also check for args_schema (Pydantic format)
        if not parameters and "args_schema" in tool:
            if hasattr(tool["args_schema"], "model_json_schema"):
                schema_output = generate_model_description(tool["args_schema"])
                parameters = schema_output.get("json_schema", {}).get("schema", {})

    return name, description, parameters


def log_tool_conversion(tool: dict[str, Any], provider: str) -> None:
    """Log tool conversion for debugging.

    Args:
        tool: The tool being converted
        provider: The provider name
    """
    try:
        name, description, parameters = extract_tool_info(tool)
        logging.debug(
            f"{provider}: Converting tool '{name}' (desc: {description[:50]}...)"
        )
        logging.debug(f"{provider}: Tool parameters: {parameters}")
    except Exception as e:
        logging.error(f"{provider}: Error extracting tool info: {e}")
        logging.error(f"{provider}: Tool structure: {tool}")


def sanitize_function_name(name: str) -> str:
    """Sanitize function name for LLM provider compatibility.

    Args:
        name: Original function name

    Returns:
        Sanitized function name (lowercase, a-z0-9_ only, max 64 chars)
    """
    return sanitize_tool_name(name)


def safe_tool_conversion(
    tool: dict[str, Any], provider: str
) -> tuple[str, str, dict[str, Any]]:
    """Safely extract and validate tool information.

    Combines extraction, validation, and logging for robust tool conversion.

    Args:
        tool: Tool dictionary to convert
        provider: Provider name for error messages and logging

    Returns:
        Tuple of (validated_name, description, parameters)

    Raises:
        ValueError: If tool is invalid or name validation fails
    """
    try:
        log_tool_conversion(tool, provider)

        name, description, parameters = extract_tool_info(tool)

        # Sanitize name before validation (replace invalid chars with underscores)
        sanitized_name = sanitize_function_name(name)

        validated_name = validate_function_name(sanitized_name, provider)

        logging.info(f"{provider}: Successfully validated tool '{validated_name}'")
        return validated_name, description, parameters
    except Exception as e:
        logging.error(f"{provider}: Error converting tool: {e}")
        raise
