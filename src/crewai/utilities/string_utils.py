import re
from typing import Any, Dict, List, Optional, Union


def interpolate_only(
    input_string: Optional[str],
    inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]],
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
    def validate_type(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, (dict, list)):
            for item in value.values() if isinstance(value, dict) else value:
                validate_type(item)
            return
        raise ValueError(
            f"Unsupported type {type(value).__name__} in inputs. "
            "Only str, int, float, bool, dict, and list are allowed."
        )

    # Validate all input values
    for key, value in inputs.items():
        try:
            validate_type(value)
        except ValueError as e:
            raise ValueError(f"Invalid value for key '{key}': {str(e)}") from e

    if input_string is None or not input_string:
        return ""
    if "{" not in input_string and "}" not in input_string:
        return input_string
    if not inputs:
        raise ValueError(
            "Inputs dictionary cannot be empty when interpolating variables"
        )

    # The regex pattern to find valid variable placeholders
    # Matches {variable_name} where variable_name starts with a letter/underscore
    # and contains only letters, numbers, and underscores
    pattern = r"\{([A-Za-z_][A-Za-z0-9_]*)\}"

    # Find all matching variables in the input string
    variables = re.findall(pattern, input_string)
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


from typing import Optional


def sanitize_collection_name(name: Optional[str]) -> str:
    """
    Sanitize a collection name to meet ChromaDB requirements:
    1. 3-63 characters long
    2. Starts and ends with alphanumeric character
    3. Contains only alphanumeric characters, underscores, or hyphens
    4. No consecutive periods
    5. Not a valid IPv4 address

    Args:
        name: The original collection name to sanitize

    Returns:
        A sanitized collection name that meets ChromaDB requirements
    """
    if not name:
        return "default_collection"

    # Replace spaces and invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    # Ensure it starts with alphanumeric
    if not sanitized[0].isalnum():
        sanitized = "a" + sanitized

    # Ensure it ends with alphanumeric
    if not sanitized[-1].isalnum():
        sanitized = sanitized[:-1] + "z"

    # Ensure length is between 3-63 characters
    if len(sanitized) < 3:
        # Add padding with alphanumeric character at the end
        sanitized = sanitized + "x" * (3 - len(sanitized))
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
        # Ensure it still ends with alphanumeric after truncation
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + "z"

    return sanitized
