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
    config = values.get("config", {})
    if not config:
        return values

    # Copy values from config (originally from YAML) to the model's attributes.
    # Only copy if the attribute isn't already set, preserving any explicitly defined values.
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

    # Remove the config from values to avoid duplicate processing
    values.pop("config", None)
    return values
