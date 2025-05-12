import os
from typing import Any

from crewai.cli.constants import DEFAULT_LLM_MODEL, ENV_VARS, LITELLM_PARAMS
from crewai.llm import LLM, BaseLLM


def create_llm(
    llm_value: str | LLM | Any | None = None,
) -> LLM | BaseLLM | None:
    """Creates or returns an LLM instance based on the given llm_value.

    Args:
        llm_value (str | BaseLLM | Any | None):
            - str: The model name (e.g., "gpt-4").
            - BaseLLM: Already instantiated BaseLLM (including LLM), returned as-is.
            - Any: Attempt to extract known attributes like model_name, temperature, etc.
            - None: Use environment-based or fallback default model.

    Returns:
        A BaseLLM instance if successful, or None if something fails.

    """
    # 1) If llm_value is already a BaseLLM or LLM object, return it directly
    if isinstance(llm_value, (LLM, BaseLLM)):
        return llm_value

    # 2) If llm_value is a string (model name)
    if isinstance(llm_value, str):
        try:
            return LLM(model=llm_value)
        except Exception:
            return None

    # 3) If llm_value is None, parse environment variables or use default
    if llm_value is None:
        return _llm_via_environment_or_fallback()

    # 4) Otherwise, attempt to extract relevant attributes from an unknown object
    try:
        # Extract attributes with explicit types
        model = (
            getattr(llm_value, "model", None)
            or getattr(llm_value, "model_name", None)
            or getattr(llm_value, "deployment_name", None)
            or str(llm_value)
        )
        temperature: float | None = getattr(llm_value, "temperature", None)
        max_tokens: int | None = getattr(llm_value, "max_tokens", None)
        logprobs: int | None = getattr(llm_value, "logprobs", None)
        timeout: float | None = getattr(llm_value, "timeout", None)
        api_key: str | None = getattr(llm_value, "api_key", None)
        base_url: str | None = getattr(llm_value, "base_url", None)
        api_base: str | None = getattr(llm_value, "api_base", None)

        return LLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
            api_base=api_base,
        )
    except Exception:
        return None


def _llm_via_environment_or_fallback() -> LLM | None:
    """Helper function: if llm_value is None, we load environment variables or fallback default model."""
    model_name = (
        os.environ.get("MODEL")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("OPENAI_MODEL_NAME")
        or DEFAULT_LLM_MODEL
    )

    # Initialize parameters with correct types
    model: str = model_name
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    logprobs: int | None = None
    timeout: float | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    logit_bias: dict[int, float] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    top_logprobs: int | None = None
    callbacks: list[Any] = []

    # Optional base URL from env
    base_url = (
        os.environ.get("BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )

    api_base = os.environ.get("API_BASE") or os.environ.get("AZURE_API_BASE")

    # Synchronize base_url and api_base if one is populated and the other is not
    if base_url and not api_base:
        api_base = base_url
    elif api_base and not base_url:
        base_url = api_base

    # Initialize llm_params dictionary
    llm_params: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_completion_tokens,
        "logprobs": logprobs,
        "timeout": timeout,
        "api_key": api_key,
        "base_url": base_url,
        "api_base": api_base,
        "api_version": api_version,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "top_p": top_p,
        "n": n,
        "stop": stop,
        "logit_bias": logit_bias,
        "response_format": response_format,
        "seed": seed,
        "top_logprobs": top_logprobs,
        "callbacks": callbacks,
    }

    UNACCEPTED_ATTRIBUTES = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION_NAME",
    ]
    set_provider = model_name.split("/")[0] if "/" in model_name else "openai"

    if set_provider in ENV_VARS:
        env_vars_for_provider = ENV_VARS[set_provider]
        if isinstance(env_vars_for_provider, (list, tuple)):
            for env_var in env_vars_for_provider:
                key_name = env_var.get("key_name")
                if key_name and key_name not in UNACCEPTED_ATTRIBUTES:
                    env_value = os.environ.get(key_name)
                    if env_value:
                        # Map environment variable names to recognized parameters
                        param_key = _normalize_key_name(key_name.lower())
                        llm_params[param_key] = env_value
                elif isinstance(env_var, dict):
                    if env_var.get("default", False):
                        for key, value in env_var.items():
                            if key not in ["prompt", "key_name", "default"]:
                                llm_params[key.lower()] = value
                else:
                    pass

    # Remove None values
    llm_params = {k: v for k, v in llm_params.items() if v is not None}

    # Try creating the LLM
    try:
        return LLM(**llm_params)
    except Exception:
        return None


def _normalize_key_name(key_name: str) -> str:
    """Maps environment variable names to recognized litellm parameter keys,
    using patterns from LITELLM_PARAMS.
    """
    for pattern in LITELLM_PARAMS:
        if pattern in key_name:
            return pattern
    return key_name
