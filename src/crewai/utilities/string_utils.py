import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from crewai.utilities.jinja_templating import render_template


def interpolate_only(
    input_string: Optional[str],
    inputs: Dict[str, Any],
) -> str:
    """Interpolate placeholders (e.g., {key}) in a string while leaving JSON untouched.
    Only interpolates placeholders that follow the pattern {variable_name} where
    variable_name starts with a letter/underscore and contains only letters, numbers, and underscores.
    
    This function now supports advanced Jinja2 templating features:
    - Container types (List, Dict, Set)
    - Standard objects (datetime, time)
    - Custom objects
    - Conditional and loop statements
    - Filtering options

    Args:
        input_string: The string containing template variables to interpolate.
                     Can be None or empty, in which case an empty string is returned.
        inputs: Dictionary mapping template variables to their values.
               Supports all types of values including complex objects.

    Returns:
        The interpolated string with all template variables replaced with their values.
        Empty string if input_string is None or empty.

    Raises:
        ValueError: If inputs dictionary is empty when interpolating variables.
        KeyError: If a required template variable is missing from inputs.
    """
    def validate_type(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, (dict, list)):
            for item in value.values() if isinstance(value, dict) else value:
                validate_type(item)
            return
        if isinstance(value, datetime):
            return
        # Check if it's a Pydantic model or other known custom type
        try:
            from pydantic import BaseModel
            if isinstance(value, BaseModel):
                return
        except ImportError:
            pass
            
        raise ValueError(
            f"Unsupported type {type(value).__name__} in inputs. "
            "Only str, int, float, bool, dict, list, datetime, and custom objects are allowed."
        )

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

    # Check if the template contains Jinja2 syntax ({% ... %} or {{ ... }})
    has_jinja_syntax = "{{" in input_string or "{%" in input_string
    has_complex_indexing = re.search(r"\{([A-Za-z_][A-Za-z0-9_]*)\[[0-9]+\]\}", input_string)

    if has_jinja_syntax or has_complex_indexing:
        return render_template(input_string, inputs)
    else:
        # The regex pattern to find valid variable placeholders
        # Matches {variable_name} where variable_name starts with a letter/underscore
        # and contains only letters, numbers, and underscores
        pattern = r"\{([A-Za-z_][A-Za-z0-9_]*)\}"

        # Find all matching variables in the input string
        variables = re.findall(pattern, input_string)
        
        # Check if all variables exist in inputs
        missing_vars = [var for var in variables if var not in inputs]
        if missing_vars:
            raise KeyError(
                f"Template variable '{missing_vars[0]}' not found in inputs dictionary"
            )
        
        try:
            return render_template(input_string, inputs)
        except Exception:
            result = input_string
            for var in variables:
                if var in inputs:
                    placeholder = "{" + var + "}"
                    value = str(inputs[var])
                    result = result.replace(placeholder, value)
            return result
