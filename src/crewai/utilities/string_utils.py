import re
from typing import Any, Dict, Optional, Union

# Regex to parse chained accessors like .attribute, [0], ['key'], ["key"]
# Ensures a consistent number of capture groups regardless of match type.
# Group 1: Full dot accessor (e.g., `.attr`)
# Group 2: Attribute name (e.g., `attr`)
# Group 3: Full bracket accessor (e.g., `[0]`, `['key']`)
# Group 4: Content inside brackets (e.g., `0`, `'key'`, `"key"`)
# Group 5: Integer index (e.g., `0`)
# Group 6: Single-quoted key content (e.g., `key`) - without quotes
# Group 7: Double-quoted key content (e.g., `key`) - without quotes
_ACCESSOR_PATTERN = re.compile(
    # Option 1: Dot Accessor
    r"(\.\s*([A-Za-z_][A-Za-z0-9_]*))"  # Groups 1 (full), 2 (name)
    # Option 2: Bracket Accessor
    r"|(\[\s*("  # Groups 3 (full), 4 (content w/ quotes)
    #   Content Type 1: Integer Index
    r"([0-9]+)"  # Group 5 (index)
    #   Content Type 2: Single-Quoted Key
    r"|'((?:[^'\\]|\\.)*)'"  # Group 6 (key content)
    #   Content Type 3: Double-Quoted Key
    r"|\"((?:[^\"\\]|\\.)*)\""  # Group 7 (key content)
    r")\s*\])"
)


def _evaluate_accessors(base_value: Any, accessor_string: str) -> Any:
    """
    Safely evaluate a chain of attribute and item accessors on a base value.

    Args:
        base_value: The initial value (e.g., from the inputs dictionary).
        accessor_string: A string representing chained accessors like
                         '.attribute', '[0]', '['key']', '.attr[0]['key']'.
                         Whitespace around dots and brackets is ignored.

    Returns:
        The final value after applying all accessors.

    Raises:
        KeyError: If a dictionary key or list index is not found or invalid.
        AttributeError: If an attribute is not found.
        TypeError: If an object is not subscriptable or doesn't support attribute access
                   at the point the accessor is applied.
        ValueError: If the accessor string format is invalid or cannot be parsed.
        RuntimeError: For other unexpected errors during evaluation.
    """
    current_value = base_value
    remaining_accessor_string = accessor_string.strip()

    while remaining_accessor_string:
        match = _ACCESSOR_PATTERN.match(remaining_accessor_string)
        if not match:
            raise ValueError(
                f"Invalid accessor format near '{remaining_accessor_string}' "
                f"in full accessor string '{accessor_string}'"
            )

        # Unpack all 7 potential groups - some will be None depending on match
        (
            dot_full,
            attr_name,
            bracket_full,
            bracket_content_raw,
            index_str,
            single_quoted_key,
            double_quoted_key,
        ) = match.groups()
        full_match_str = match.group(0)  # The actual matched string portion

        try:
            if dot_full:  # Matched dot accessor (Group 1 is not None)
                # Attribute access (.attribute) - Use attr_name (Group 2)
                if isinstance(current_value, dict):
                    try:
                        # Try dict key access first
                        current_value = current_value[attr_name]
                    except KeyError:
                        # Fallback to attribute access on the dict itself or contained object
                        current_value = getattr(current_value, attr_name)
                else:
                    # Standard attribute access for non-dicts
                    current_value = getattr(current_value, attr_name)

            else:  # Matched bracket accessor (bracket_full / Group 3 is not None)
                key: Union[str, int]
                if index_str:  # Matched integer index (Group 5 is not None)
                    key = int(index_str)
                    current_value = current_value[key]
                else:  # Matched string key (Group 6 or 7 is not None)
                    # Determine key content from single (Group 6) or double (Group 7) quoted group
                    raw_key = (
                        single_quoted_key
                        if single_quoted_key is not None
                        else double_quoted_key
                    )
                    # Handle potential escapes within the key string
                    key = raw_key.encode().decode("unicode_escape")
                    current_value = current_value[key]

        # --- Error Handling specific to the type of access ---
        except AttributeError:
            # Error message should use attr_name if dot access failed
            failed_name = attr_name if dot_full else "attribute"
            raise AttributeError(
                f"Attribute '{failed_name}' not found on object "
                f"of type {type(current_value).__name__} while evaluating "
                f"'{accessor_string}'"
            ) from None  # Use None to break chain, more direct error
        except KeyError:
            # Determine key_repr based on what matched for better error messages
            if index_str:
                key_repr = index_str
                error_type = "Index"
            else:
                # Use bracket_content_raw (Group 4) which includes quotes, or reconstruct from key
                key_repr = repr(key) if "key" in locals() else bracket_content_raw
                error_type = "Key"
            raise KeyError(
                f"{error_type} {key_repr} not found or invalid for object "
                f"of type {type(current_value).__name__} while evaluating "
                f"'{accessor_string}'"
            ) from None
        except IndexError:  # More specific than KeyError for sequences
            # Use index_str (Group 5) for the message if available
            idx_val = index_str if index_str else "unknown"
            raise KeyError(  # Raising KeyError for consistency with original tests
                f"Index {idx_val} out of bounds for sequence of length {len(current_value)} "
                f"while evaluating '{accessor_string}'"
            ) from None
        except TypeError as e:
            # E.g., trying to index a non-subscriptable object or getattr on non-object
            raise TypeError(
                f"Object of type {type(current_value).__name__} does not support "
                f"{'attribute' if dot_full else 'item'} access "
                f"needed for '{full_match_str}' in '{accessor_string}': {e}"
            ) from None
        except Exception as e:  # Catch unexpected errors during access
            raise RuntimeError(
                f"Unexpected error accessing '{full_match_str}' in "
                f"'{accessor_string}': {type(e).__name__}: {e}"
            ) from e

        # Move to the next part of the accessor string
        remaining_accessor_string = remaining_accessor_string[match.end() :].lstrip()

    return current_value


def interpolate_only(
    input_string: Optional[str],
    inputs: Dict[str, Any],  # Allow Any type, validation happens during access
) -> str:
    """
    Interpolates placeholders in a string using values from a dictionary,
    handling nested attribute and item access, while leaving JSON-like
    structures untouched.

    Placeholders follow the format {variable_name.accessor[index]['key']...}.
    - `variable_name` must start with a letter or underscore, followed by
      letters, numbers, or underscores.
    - Accessors (`.`, `[]`) allow navigating nested objects and lists/dicts.

    Args:
        input_string: The string containing template variables. Returns "" if
                      None or empty.
        inputs: Dictionary mapping base variable names to their values.

    Returns:
        The interpolated string.

    Raises:
        KeyError: If a base template variable (e.g., {variable}) or a key/index
                  during accessor evaluation is not found.
        AttributeError: If an attribute accessed via dot notation is not found.
        TypeError: If an object does not support the required access method
                   (e.g., indexing a non-subscriptable object).
        ValueError: If the accessor format within a placeholder is invalid,
                    or if `inputs` is empty when placeholders exist, or if
                    a base variable references an object that cannot be directly
                    converted to a string (and has no accessors).
        RuntimeError: For other unexpected errors during evaluation.
    """
    if not input_string:
        return ""
    # Optimization: Quick check for presence of { and }
    if not ("{" in input_string and "}" in input_string):
        return input_string

    # Regex to find placeholders like {var}, {var.attr}, {var[0]}, {var['key']}
    # Group 1: Base variable name (e.g., "var")
    # Group 2: The chain of accessors (e.g., ".attr[0]['key']") - may include whitespace
    placeholder_pattern = re.compile(
        r"\{"
        r"([A-Za-z_][A-Za-z0-9_]*)"  # 1: Base variable name
        r"((?:\s*(?:\.|\[).*?)*?)"  # 2: Accessors (dot or bracket), non-greedy, allows whitespace
        r"\s*\}"  # Allow whitespace before closing brace
    )

    # Use a set to collect missing base variables for a single final error
    missing_base_vars = set()
    # Use a list to store parts of the final string
    result_parts = []
    last_end = 0
    found_placeholder = False  # Flag to check if any placeholder was encountered

    for match in placeholder_pattern.finditer(input_string):
        found_placeholder = (
            True  # Mark that we found at least one potential placeholder
        )
        start, end = match.span()
        # Add the text segment before the current match
        result_parts.append(input_string[last_end:start])

        full_placeholder = match.group(0)  # e.g., { list_variable [ 0 ] }
        base_var = match.group(1)  # e.g., list_variable
        accessors = match.group(
            2
        ).strip()  # e.g., [ 0 ] -> "[0]" or . attribute -> ".attribute"

        # Check for empty inputs dict only if we actually find a placeholder
        if not inputs:
            raise ValueError(
                "Inputs dictionary cannot be empty when template variables "
                f"like '{full_placeholder}' are present."
            )

        if base_var not in inputs:
            # Base variable is missing, record it and keep the placeholder
            missing_base_vars.add(base_var)
            result_parts.append(full_placeholder)
        else:
            # Base variable exists, try to evaluate
            try:
                value = inputs[base_var]
                if accessors:
                    # Pass the potentially complex accessor string to the evaluator
                    value = _evaluate_accessors(value, accessors)

                # Check if the final value is directly interpolatable
                # Allow basic types, None. Disallow others *unless* accessed.
                if not accessors and not isinstance(
                    value, (str, int, float, bool, type(None))
                ):
                    # Special case: allow lists/dicts even without accessors, as they have default string representations
                    if not isinstance(value, (list, dict)):
                        raise ValueError(
                            f"Variable '{base_var}' resolved to an unsupported type "
                            f"({type(value).__name__}) for direct interpolation without accessors. "
                            f"Use accessors (e.g., {base_var}.attribute) or ensure "
                            f"the value is a primitive type, list, or dict."
                        )

                # Convert the final evaluated value to string for joining
                result_parts.append(str(value))

            except (
                KeyError,
                IndexError,
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as e:
                # Evaluation failed (e.g., bad index/key/attribute, invalid accessor)
                # Re-raise the specific error caught by _evaluate_accessors or the ValueError from above
                # Add context about the placeholder being processed.
                # Modify the exception args to prepend context.
                error_args = list(e.args)
                # Ensure error_args is not empty before modifying
                if error_args:
                    error_args[0] = (
                        f"Error evaluating placeholder {full_placeholder}: {error_args[0]}"
                    )
                else:
                    # Add a generic message if args was empty
                    error_args.append(
                        f"Error evaluating placeholder {full_placeholder}: {type(e).__name__}"
                    )
                e.args = tuple(error_args)
                raise e  # Re-raise the original exception type with modified message
            except Exception as e:  # Catch any other unexpected errors
                raise RuntimeError(
                    f"Unexpected error evaluating placeholder {full_placeholder}: "
                    f"{type(e).__name__}: {e}"
                ) from e

        last_end = end  # Update position for the next segment

    # Add any remaining text after the last placeholder
    result_parts.append(input_string[last_end:])

    # After checking all placeholders, raise error if any base variables were missing
    if missing_base_vars:
        raise KeyError(
            f"Template variable(s) {{{', '.join(sorted(list(missing_base_vars)))}}} "
            f"not found in inputs."
        )

    # If no placeholders were found by the regex, return the original string
    # (This handles cases where {} exist but don't match the variable pattern)
    if not found_placeholder and not missing_base_vars:
        return input_string

    return "".join(result_parts)
