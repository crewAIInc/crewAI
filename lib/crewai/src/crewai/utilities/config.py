from typing import Any

from pydantic import BaseModel


def process_config(
    values: dict[str, Any], model_class: type[BaseModel]
) -> dict[str, Any]:
    """Process the config dictionary and update the values accordingly.

    Args:
        values: The dictionary of values to update.
        model_class: The Pydantic model class to reference for field validation.

    Returns:
        The updated values dictionary.
    """
    # Handle non-dict inputs gracefully - let Pydantic handle type validation downstream
    if not isinstance(values, dict):
        return values

    config = values.get("config", {})
    if not config:
        return values

    for key, value in config.items():
        if key not in model_class.model_fields or values.get(key) is not None:
            continue

        if isinstance(value, dict):
            if isinstance(values.get(key), dict):
                values[key].update(value)
            else:
                values[key] = value
        else:
            values[key] = value

    values.pop("config", None)
    return values
