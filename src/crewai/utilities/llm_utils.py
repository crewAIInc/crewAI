import os
from typing import Any, Dict, List, Optional, Union

from crewai.cli.constants import DEFAULT_LLM_MODEL, ENV_VARS, LITELLM_PARAMS
from crewai.llm import LLM


def is_ollama_model(model_name: str) -> bool:
    """Check if a model name is an Ollama model."""
    return bool(model_name and "ollama" in str(model_name).lower())


def validate_and_set_endpoint(endpoint: str) -> str:
    """Validate and format an endpoint URL."""
    if not endpoint:
        return endpoint
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Invalid endpoint URL: {endpoint}. Must start with http:// or https://")
    return endpoint.rstrip("/")


def create_llm(
    llm_value: Union[str, LLM, Any, None] = None,
    endpoint: Optional[str] = None,
) -> Optional[LLM]:
    """
    Creates or returns an LLM instance based on the given llm_value.

    Args:
        llm_value (str | LLM | Any | None):
            - str: Model name to use for instantiating a new LLM.
            - LLM: Already instantiated LLM, returned as-is.
            - Any: Attempt to extract known attributes like model_name, temperature, etc.
            - None: Use environment-based or fallback default model.
        endpoint (str | None):
            - Optional API endpoint URL. When provided for Ollama models,
              this will override the default API endpoint. Should be a valid
              HTTP/HTTPS URL (e.g., "http://localhost:11434").

    Returns:
        An LLM instance if successful, or None if something fails.
    """

    # 1) If llm_value is already an LLM object, return it directly
    if isinstance(llm_value, LLM):
        return llm_value

    # 2) If llm_value is a string (model name)
    if isinstance(llm_value, str):
        try:
            # If endpoint is provided and model is Ollama, use it as api_base
            if endpoint and is_ollama_model(llm_value):
                created_llm = LLM(model=llm_value, api_base=validate_and_set_endpoint(endpoint))
            else:
                created_llm = LLM(model=llm_value)
            return created_llm
        except Exception as e:
            print(f"Failed to instantiate LLM with model='{llm_value}': {e}")
            return None

    # 3) If llm_value is None, parse environment variables or use default
    if llm_value is None:
        return _llm_via_environment_or_fallback(endpoint=endpoint)

    # 4) Otherwise, attempt to extract relevant attributes from an unknown object
    try:
        # Extract attributes with explicit types
        model = (
            getattr(llm_value, "model_name", None)
            or getattr(llm_value, "deployment_name", None)
            or str(llm_value)
        )
        temperature: Optional[float] = getattr(llm_value, "temperature", None)
        max_tokens: Optional[int] = getattr(llm_value, "max_tokens", None)
        logprobs: Optional[int] = getattr(llm_value, "logprobs", None)
        timeout: Optional[float] = getattr(llm_value, "timeout", None)
        api_key: Optional[str] = getattr(llm_value, "api_key", None)
        base_url: Optional[str] = getattr(llm_value, "base_url", None)
        api_base: Optional[str] = getattr(llm_value, "api_base", None)

        # If endpoint is provided and model is Ollama, use it as api_base
        if endpoint and is_ollama_model(model):
            api_base = validate_and_set_endpoint(endpoint)

        created_llm = LLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
            api_base=api_base,
        )
        return created_llm
    except Exception as e:
        print(f"Error instantiating LLM from unknown object type: {e}")
        return None


def _llm_via_environment_or_fallback(endpoint: Optional[str] = None) -> Optional[LLM]:
    """
    Helper function: if llm_value is None, we load environment variables or fallback default model.
    
    Args:
        endpoint (str | None):
            - Optional endpoint URL for the LLM API.
    """
    model_name = (
        os.environ.get("OPENAI_MODEL_NAME")
        or os.environ.get("MODEL")
        or DEFAULT_LLM_MODEL
    )

    # Initialize parameters with correct types
    model: str = model_name
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    logprobs: Optional[int] = None
    timeout: Optional[float] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    logit_bias: Optional[Dict[int, float]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    top_logprobs: Optional[int] = None
    callbacks: List[Any] = []

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
    llm_params: Dict[str, Any] = {
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
                    print(
                        f"Expected env_var to be a dictionary, but got {type(env_var)}"
                    )

    # Remove None values
    llm_params = {k: v for k, v in llm_params.items() if v is not None}

    # If endpoint is provided and model is Ollama, use it as api_base
    if endpoint and is_ollama_model(model_name):
        llm_params["api_base"] = validate_and_set_endpoint(endpoint)

    # Try creating the LLM
    try:
        new_llm = LLM(**llm_params)
        return new_llm
    except Exception as e:
        print(
            f"Error instantiating LLM from environment/fallback: {type(e).__name__}: {e}"
        )
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
