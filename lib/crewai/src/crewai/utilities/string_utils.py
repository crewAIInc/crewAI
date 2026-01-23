# sanitize_tool_name adapted from python-slugify by Val Neekman
# https://github.com/un33k/python-slugify
# MIT License

import re
from typing import Any, Final
import unicodedata


_VARIABLE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{([A-Za-z_][A-Za-z0-9_\-]*)}")
_QUOTE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\'\"]+")
_CAMEL_LOWER_UPPER: Final[re.Pattern[str]] = re.compile(r"([a-z])([A-Z])")
_CAMEL_UPPER_LOWER: Final[re.Pattern[str]] = re.compile(r"([A-Z]+)([A-Z][a-z])")
_DISALLOWED_CHARS_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9]+")
_DUPLICATE_UNDERSCORE_PATTERN: Final[re.Pattern[str]] = re.compile(r"_+")
_MAX_TOOL_NAME_LENGTH: Final[int] = 64


def sanitize_tool_name(name: str, max_length: int = _MAX_TOOL_NAME_LENGTH) -> str:
    """Sanitize tool name for LLM provider compatibility.

    Normalizes Unicode, splits camelCase, lowercases, replaces invalid characters
    with underscores, and truncates to max_length. Conforms to OpenAI/Bedrock requirements.

    Args:
        name: Original tool name.
        max_length: Maximum allowed length (default 64 per OpenAI/Bedrock limits).

    Returns:
        Sanitized tool name (lowercase, a-z0-9_ only, max 64 chars).
    """
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = _CAMEL_UPPER_LOWER.sub(r"\1_\2", name)
    name = _CAMEL_LOWER_UPPER.sub(r"\1_\2", name)
    name = name.lower()
    name = _QUOTE_PATTERN.sub("", name)
    name = _DISALLOWED_CHARS_PATTERN.sub("_", name)
    name = _DUPLICATE_UNDERSCORE_PATTERN.sub("_", name)
    name = name.strip("_")

    if len(name) > max_length:
        name = name[:max_length].rstrip("_")

    return name


def interpolate_only(
    input_string: str | None,
    inputs: dict[str, str | int | float | dict[str, Any] | list[Any]],
) -> str:
    """Interpolate placeholders (e.g., {key}) in a string while leaving JSON untouched.
    Only interpolates placeholders that follow the pattern {variable_name} where
    variable_name starts with a letter/underscore and contains only letters, numbers, and underscores.

    Args:
        input_string: The string containing template variables to interpolate.
                     Can be None or empty, in which case an empty string is returned.
        inputs: Dictionary mapping template variables to their values.
               Supported value types are strings, integers, floats, and dicts/lists
               containing only these types and other nested dicts/lists.

    Returns:
        The interpolated string with all template variables replaced with their values.
        Empty string if input_string is None or empty.

    Raises:
        ValueError: If a value contains unsupported types or a template variable is missing
    """

    # Validation function for recursive type checking
    def _validate_type(validate_value: Any) -> None:
        if validate_value is None:
            return
        if isinstance(validate_value, (str, int, float, bool)):
            return
        if isinstance(validate_value, (dict, list)):
            for item in (
                validate_value.values()
                if isinstance(validate_value, dict)
                else validate_value
            ):
                _validate_type(item)
            return
        raise ValueError(
            f"Unsupported type {type(validate_value).__name__} in inputs. "
            "Only str, int, float, bool, dict, and list are allowed."
        )

    # Validate all input values
    for key, value in inputs.items():
        try:
            _validate_type(value)
        except ValueError as e:  # noqa: PERF203
            raise ValueError(f"Invalid value for key '{key}': {e!s}") from e

    if input_string is None or not input_string:
        return ""
    if "{" not in input_string and "}" not in input_string:
        return input_string
    if not inputs:
        raise ValueError(
            "Inputs dictionary cannot be empty when interpolating variables"
        )

    variables = _VARIABLE_PATTERN.findall(input_string)
    result = input_string

    # Check if all variables exist in inputs
    missing_vars = [var for var in variables if var not in inputs]
    if missing_vars:
        raise KeyError(
            f"Template variable '{missing_vars[0]}' not found in inputs dictionary"
        )

    # Replace each variable with its value
    for var in variables:
        if var in inputs:
            placeholder = "{" + var + "}"
            value = str(inputs[var])
            result = result.replace(placeholder, value)

    return result
