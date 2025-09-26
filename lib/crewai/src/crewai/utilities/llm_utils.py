import logging
import os
from typing import Any, Final

from crewai.cli.constants import DEFAULT_LLM_MODEL, ENV_VARS, LITELLM_PARAMS
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)


def create_llm(
    llm_value: str | LLM | Any | None = None,
) -> LLM | BaseLLM | None:
    """Creates or returns an LLM instance based on the given llm_value.

    Args:
        llm_value: LLM instance, model name string, None, or an object with LLM attributes.

    Returns:
        A BaseLLM instance if successful, or None if something fails.
    """

    if isinstance(llm_value, (LLM, BaseLLM)):
        return llm_value

    if isinstance(llm_value, str):
        try:
            return LLM(model=llm_value)
        except Exception as e:
            logger.debug(f"Failed to instantiate LLM with model='{llm_value}': {e}")
            return None

    if llm_value is None:
        return _llm_via_environment_or_fallback()

    try:
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
    except Exception as e:
        logger.debug(f"Error instantiating LLM from unknown object type: {e}")
        return None


UNACCEPTED_ATTRIBUTES: Final[list[str]] = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION_NAME",
]


def _llm_via_environment_or_fallback() -> LLM | None:
    """Creates an LLM instance based on environment variables or defaults.

    Returns:
        A BaseLLM instance if successful, or None if something fails.
    """
    model_name = (
        os.environ.get("MODEL")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("OPENAI_MODEL_NAME")
        or DEFAULT_LLM_MODEL
    )

    model: str = model_name
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    logprobs: int | None = None
    timeout: float | None = None
    api_key: str | None = None
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
                    logger.debug(
                        f"Expected env_var to be a dictionary, but got {type(env_var)}"
                    )

    llm_params = {k: v for k, v in llm_params.items() if v is not None}

    try:
        return LLM(**llm_params)
    except Exception as e:
        logger.debug(
            f"Error instantiating LLM from environment/fallback: {type(e).__name__}: {e}"
        )
        return None


def _normalize_key_name(key_name: str) -> str:
    """Maps environment variable names to recognized litellm parameter keys.

    Args:
        key_name: The environment variable name to normalize.
    """
    for pattern in LITELLM_PARAMS:
        if pattern in key_name:
            return pattern
    return key_name
