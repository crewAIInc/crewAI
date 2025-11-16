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
    from crewai.utilities.constants import NOT_SPECIFIED

    config = values.get("config", {})
    if not config:
        return values

    for key, value in config.items():
        if key not in model_class.model_fields:
            continue

        current = values.get(key)

        if current is not None and current is not NOT_SPECIFIED:
            continue

        if isinstance(value, dict):
            if isinstance(current, dict):
                current.update(value)
            else:
                values[key] = value
        else:
            values[key] = value

    values.pop("config", None)
    return values
