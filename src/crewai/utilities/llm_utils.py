import os
from typing import Any, Dict, Optional, Union

from packaging import version

from crewai.cli.constants import ENV_VARS, LITELLM_PARAMS
from crewai.cli.utils import read_toml
from crewai.cli.version import get_crewai_version
from crewai.llm import LLM


def create_llm(
    llm_value: Union[str, LLM, Any, None] = None,
    default_model: str = "gpt-4o-mini",
) -> Optional[LLM]:
    """
    Creates or returns an LLM instance based on the given llm_value.

    Args:
        llm_value (str | LLM | Any | None):
            - str: The model name (e.g., "gpt-4").
            - LLM: Already instantiated LLM, returned as-is.
            - Any: Attempt to extract known attributes like model_name, temperature, etc.
            - None: Use environment-based or fallback default model.
        default_model (str): The fallback model name to use if llm_value is None
                             and no environment variable is set.

    Returns:
        An LLM instance if successful, or None if something fails.
    """

    # 1) If llm_value is already an LLM object, return it directly
    if isinstance(llm_value, LLM):
        return llm_value

    # 2) If llm_value is a string (model name)
    if isinstance(llm_value, str):
        try:
            created_llm = LLM(model=llm_value)
            print(f"LLM created with model='{llm_value}'")
            return created_llm
        except Exception as e:
            print(f"Failed to instantiate LLM with model='{llm_value}': {e}")
            return None

    # 3) If llm_value is None, parse environment variables or use default
    if llm_value is None:
        return _llm_via_environment_or_fallback(default_model)

    # 4) Otherwise, attempt to extract relevant attributes from an unknown object (like a config)
    #    e.g. follow the approach used in agent.py
    try:
        llm_params = {
            "model": (
                getattr(llm_value, "model_name", None)
                or getattr(llm_value, "deployment_name", None)
                or str(llm_value)
            ),
            "temperature": getattr(llm_value, "temperature", None),
            "max_tokens": getattr(llm_value, "max_tokens", None),
            "logprobs": getattr(llm_value, "logprobs", None),
            "timeout": getattr(llm_value, "timeout", None),
            "max_retries": getattr(llm_value, "max_retries", None),
            "api_key": getattr(llm_value, "api_key", None),
            "base_url": getattr(llm_value, "base_url", None),
            "organization": getattr(llm_value, "organization", None),
        }
        # Remove None values
        llm_params = {k: v for k, v in llm_params.items() if v is not None}
        created_llm = LLM(**llm_params)
        print(
            "LLM created with extracted parameters; "
            f"model='{llm_params.get('model', 'UNKNOWN')}'"
        )
        return created_llm
    except Exception as e:
        print(f"Error instantiating LLM from unknown object type: {e}")
        return None


def create_chat_llm(default_model: str = "gpt-4") -> Optional[LLM]:
    """
    Creates a Chat LLM with additional checks, such as verifying crewAI version
    or reading from pyproject.toml. Then calls `create_llm(None, default_model)`.

    Args:
        default_model (str): Fallback model if not set in environment.

    Returns:
        An instance of LLM or None if instantiation fails.
    """
    print("[create_chat_llm] Checking environment and version info...")

    crewai_version = get_crewai_version()
    min_required_version = "0.87.0"  # Update to latest if needed

    pyproject_data = read_toml()
    if pyproject_data.get("tool", {}).get("poetry") and (
        version.parse(crewai_version) < version.parse(min_required_version)
    ):
        print(
            f"You are running an older version of crewAI ({crewai_version}) that uses poetry.\n"
            "Please run `crewai update` to switch to uv-based builds."
        )

    # After checks, simply call create_llm with None (meaning "use env or fallback"):
    return create_llm(None, default_model=default_model)


def _llm_via_environment_or_fallback(
    default_model: str = "gpt-4o-mini",
) -> Optional[LLM]:
    """
    Helper function: if llm_value is None, we load environment variables or fallback default model.
    """
    model_name = (
        os.environ.get("OPENAI_MODEL_NAME") or os.environ.get("MODEL") or default_model
    )
    llm_params = {"model": model_name}

    # Optional base URL from env
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if api_base:
        llm_params["base_url"] = api_base

    UNACCEPTED_ATTRIBUTES = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION_NAME",
    ]
    set_provider = model_name.split("/")[0] if "/" in model_name else "openai"

    if set_provider in ENV_VARS:
        for env_var in ENV_VARS[set_provider]:
            key_name = env_var.get("key_name")
            if key_name and key_name not in UNACCEPTED_ATTRIBUTES:
                env_value = os.environ.get(key_name)
                if env_value:
                    # Map environment variable names to recognized LITELLM_PARAMS if any
                    param_key = _normalize_key_name(key_name.lower())
                    llm_params[param_key] = env_value
            elif env_var.get("default", False):
                for key, value in env_var.items():
                    if key not in ["prompt", "key_name", "default"]:
                        if key in os.environ:
                            llm_params[key] = value

    # Try creating the LLM
    try:
        new_llm = LLM(**llm_params)
        print(f"LLM created with model='{model_name}'")
        return new_llm
    except Exception as e:
        print(f"Error instantiating LLM from environment/fallback: {e}")
        return None


def _normalize_key_name(key_name: str) -> str:
    """
    Maps environment variable names to recognized litellm parameter keys,
    using patterns from LITELLM_PARAMS.
    """
    for pattern in LITELLM_PARAMS:
        if pattern in key_name:
            return pattern
    return key_name
