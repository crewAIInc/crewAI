import os
from typing import Any, Dict, List, Optional, Union

from crewai.cli.constants import DEFAULT_LLM_MODEL, ENV_VARS, LITELLM_PARAMS
from crewai.llm import LLM, BaseLLM


def create_llm(
    llm_value: Union[str, LLM, Any, None] = None,
    prefer_native: Optional[bool] = None,
) -> Optional[LLM | BaseLLM]:
    """
    Creates or returns an LLM instance based on the given llm_value.
    Now supports provider prefixes like 'openai/gpt-4' for native implementations.

    Args:
        llm_value (str | BaseLLM | Any | None):
            - str: The model name (e.g., "gpt-4" or "openai/gpt-4").
            - BaseLLM: Already instantiated BaseLLM (including LLM), returned as-is.
            - Any: Attempt to extract known attributes like model_name, temperature, etc.
            - None: Use environment-based or fallback default model.
        prefer_native (bool | None):
            - True: Use native provider implementations when available
            - False: Always use LiteLLM implementation
            - None: Use environment variable CREWAI_PREFER_NATIVE_LLMS (default: True)
            - Note: Provider prefixes (openai/, anthropic/) override this setting

    Returns:
        A BaseLLM instance if successful, or None if something fails.

    Examples:
        create_llm("gpt-4")           # Uses LiteLLM or native based on prefer_native
        create_llm("openai/gpt-4")    # Always uses native OpenAI implementation
        create_llm("anthropic/claude-3-sonnet")  # Future: native Anthropic
    """

    # 1) If llm_value is already a BaseLLM or LLM object, return it directly
    if isinstance(llm_value, LLM) or isinstance(llm_value, BaseLLM):
        return llm_value

    # 2) Determine if we should prefer native implementations (unless provider prefix is used)
    if prefer_native is None:
        prefer_native = os.getenv("CREWAI_PREFER_NATIVE_LLMS", "true").lower() in (
            "true",
            "1",
            "yes",
        )

    # 3) If llm_value is a string (model name)
    if isinstance(llm_value, str):
        try:
            # Provider prefix (openai/, anthropic/) always takes precedence
            if "/" in llm_value:
                created_llm = LLM(model=llm_value)  # LLM class handles routing
                return created_llm

            # Try native implementation first if preferred and no prefix
            if prefer_native:
                native_llm = _create_native_llm(llm_value)
                if native_llm:
                    return native_llm

            # Fallback to LiteLLM
            created_llm = LLM(model=llm_value)
            return created_llm
        except Exception as e:
            print(f"Failed to instantiate LLM with model='{llm_value}': {e}")
            return None

    # 4) If llm_value is None, parse environment variables or use default
    if llm_value is None:
        return _llm_via_environment_or_fallback(prefer_native)

    # 5) Otherwise, attempt to extract relevant attributes from an unknown object
    try:
        # Extract attributes with explicit types
        model = (
            getattr(llm_value, "model", None)
            or getattr(llm_value, "model_name", None)
            or getattr(llm_value, "deployment_name", None)
            or str(llm_value)
        )

        # Extract other parameters
        temperature: Optional[float] = getattr(llm_value, "temperature", None)
        max_tokens: Optional[int] = getattr(llm_value, "max_tokens", None)
        logprobs: Optional[int] = getattr(llm_value, "logprobs", None)
        timeout: Optional[float] = getattr(llm_value, "timeout", None)
        api_key: Optional[str] = getattr(llm_value, "api_key", None)
        base_url: Optional[str] = getattr(llm_value, "base_url", None)
        api_base: Optional[str] = getattr(llm_value, "api_base", None)

        # Use LLM class constructor which handles routing
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


def _create_native_llm(model: str, **kwargs) -> Optional[BaseLLM]:
    """
    Create a native LLM implementation based on the model name.

    Args:
        model: The model name (e.g., 'gpt-4', 'claude-3-sonnet')
        **kwargs: Additional parameters for the LLM

    Returns:
        Native LLM instance if supported, None otherwise
    """
    try:
        # OpenAI models
        if _is_openai_model(model):
            from crewai.llms.openai import OpenAILLM

            return OpenAILLM(model=model, **kwargs)

        # Claude models
        if _is_claude_model(model):
            from crewai.llms.anthropic import ClaudeLLM

            return ClaudeLLM(model=model, **kwargs)

        # Gemini models
        if _is_gemini_model(model):
            from crewai.llms.google import GeminiLLM

            return GeminiLLM(model=model, **kwargs)

        # No native implementation found
        return None

    except Exception as e:
        print(f"Failed to create native LLM for model '{model}': {e}")
        return None


def _is_openai_model(model: str) -> bool:
    """Check if a model is from OpenAI."""
    openai_prefixes = (
        "gpt-",
        "text-davinci",
        "text-curie",
        "text-babbage",
        "text-ada",
        "davinci",
        "curie",
        "babbage",
        "ada",
        "o1-",
        "o3-",
        "o4-",
        "chatgpt-",
    )

    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in openai_prefixes)


def _is_claude_model(model: str) -> bool:
    """Check if a model is from Anthropic (Claude)."""
    claude_prefixes = (
        "claude-",
        "claude",  # For cases like just "claude"
    )

    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in claude_prefixes)


def _is_gemini_model(model: str) -> bool:
    """Check if a model is from Google (Gemini)."""
    gemini_prefixes = (
        "gemini-",
        "gemini",  # For cases like just "gemini"
    )

    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in gemini_prefixes)


def _llm_via_environment_or_fallback(
    prefer_native: bool = True,
) -> Optional[LLM | BaseLLM]:
    """
    Helper function: if llm_value is None, we load environment variables or fallback default model.
    Now with native provider support.
    """
    model_name = (
        os.environ.get("MODEL")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("OPENAI_MODEL_NAME")
        or DEFAULT_LLM_MODEL
    )

    # Try native implementation first if preferred
    if prefer_native:
        native_llm = _create_native_llm(model_name)
        if native_llm:
            return native_llm

    # Initialize parameters with correct types (original logic continues)
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
    set_provider = model_name.partition("/")[0] if "/" in model_name else "openai"

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
