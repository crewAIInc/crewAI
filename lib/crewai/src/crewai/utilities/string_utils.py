import re
from typing import Any, Final

_VARIABLE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{([A-Za-z_][A-Za-z0-9_\-]*)}")


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
