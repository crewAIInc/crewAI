from typing import Any
from pydantic import BaseModel
import copy

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

    for key, value in config.items():
        if key not in model_class.model_fields:
            continue
        if (override_value := values.get(key)) is not None:
            if isinstance(override_value, dict) and isinstance(value, dict):
                config_value = copy.deepcopy(value)
                _dict_deep_update(config_value, override_value)
                values[key] = config_value
        else:
            values[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

    values.pop("config", None)
    return values

def _dict_deep_update(to_dict: dict[str, Any], from_dict: dict[str, Any]) -> None:
    """Internal helper to recursively update to_dict with from_dict values in-place."""
    for key, value in from_dict.items():
        if key in to_dict and isinstance(to_dict[key], dict) and isinstance(value, dict):
            _dict_deep_update(to_dict[key], value)
        else:
            to_dict[key] = value
