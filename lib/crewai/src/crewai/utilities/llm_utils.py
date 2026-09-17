import logging
import os
from typing import Any, Final

from pydantic import ValidationError

from crewai.constants import DEFAULT_LLM_MODEL, ENV_VARS, LITELLM_PARAMS
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM


logger = logging.getLogger(__name__)


def create_llm(
    llm_value: str | dict[str, Any] | LLM | Any | None = None,
) -> LLM | BaseLLM | None:
    """Creates or returns an LLM instance based on the given llm_value.

    Args:
        llm_value: LLM instance, model name string, config dict, None, or an
            object with LLM attributes.

    Returns:
        A BaseLLM instance if successful, or None if something fails.
    """

    if isinstance(llm_value, (LLM, BaseLLM)):
        return llm_value

    if isinstance(llm_value, str):
        try:
            return LLM(model=llm_value)
        except Exception as e:
            logger.error(f"Error instantiating LLM from string: {e}")
            raise e

    if isinstance(llm_value, dict):
        try:
            model = (
                llm_value.get("model")
                or llm_value.get("model_name")
                or llm_value.get("deployment_name")
            )
            if not model:
                raise ValueError(
                    "LLM config dictionaries must include 'model', "
                    "'model_name', or 'deployment_name'"
                )
            llm_params = {**llm_value, "model": model}
            llm_params.pop("model_name", None)
            llm_params.pop("deployment_name", None)
            return LLM(**llm_params)
        except Exception as e:
            logger.error(f"Error instantiating LLM from dict: {e}")
            raise e

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
        max_tokens: float | int | None = getattr(llm_value, "max_tokens", None)
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
        logger.error(f"Error instantiating LLM from unknown object type: {e}")
        raise e


# Generation and runtime settings: carried to any target whose class has the
# field. Every one of them means the same thing on every provider that has it.
GENERATION_SETTINGS: Final[tuple[str, ...]] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "stop",
    "seed",
    "n",
    "timeout",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "callbacks",
    "stream",
    "prefer_upload",
)

# Carried only to a target on the same provider: credentials and endpoints,
# how the SDK client is built, and settings whose type or meaning is the
# provider's own. Another provider gets its own defaults and environment.
PROVIDER_SETTINGS: Final[tuple[str, ...]] = (
    # credentials and endpoints
    "api_key",
    "base_url",
    "api_base",
    "api_version",
    "organization",
    "project",
    "endpoint",
    "credential_scopes",
    "location",
    "use_vertexai",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "region_name",
    # client construction
    "max_retries",
    "default_headers",
    "default_query",
    "client_params",
    "interceptor",
    # provider-typed settings
    "response_format",
    "thinking",
    "tool_search",
    "safety_settings",
    "guardrail_config",
    "additional_model_request_fields",
    "additional_model_response_field_paths",
    "tools",
    "api",
    "instructions",
    "store",
    "include",
    "builtin_tools",
    "parse_tool_outputs",
    "auto_chain",
    "auto_chain_reasoning",
    "additional_params",
)

# Settings a provider fills from the model when the caller did not: Anthropic
# sets ``max_tokens`` to the model's output cap. The new model derives its own.
_MODEL_DERIVED: Final[frozenset[str]] = frozenset({"max_tokens"})

# The names providers give the output-token cap. A cap configured under one
# keeps its meaning under another when the target only has that one.
_CAP_NAMES: Final[tuple[str, ...]] = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
)

# Azure bakes the declared model's deployment into the endpoint at construction
# (``https://<r>.openai.azure.com/openai/deployments/<model>``). The new model
# must get its own deployment, so only the resource root is carried.
_AZURE_DEPLOYMENT_PATH: Final[str] = "/openai/deployments/"

# The declared endpoint, under the names the classes give it.
_ENDPOINT_NAMES: Final[tuple[str, ...]] = ("base_url", "api_base")


def create_llm_like(model: str, base: BaseLLM | None) -> BaseLLM:
    """Build ``model`` configured like ``base``.

    The instance is built through :class:`LLM`, so provider routing and every
    setting derived from the model (context window, reasoning flags, the SDK
    client, Azure's deployment path) are computed for the new model; what comes
    from ``base`` is the configuration a caller put on it. :data:`GENERATION_SETTINGS`
    go to any target whose class has the field. :data:`PROVIDER_SETTINGS` —
    credentials, endpoints, client construction, provider-typed settings — go
    only to a target on the same provider as ``base`` (:func:`_same_provider`):
    a swap across providers must not send, say, an OpenAI key and proxy URL to
    Anthropic. A value ``base``'s provider derived from its model rather than
    took from the caller is not carried either (see :data:`_MODEL_DERIVED`), nor
    are ``additional_params`` across classes (they are the class's own extra
    kwargs). A declared llm that runs through LiteLLM swaps to LiteLLM, whatever
    the new model's native class would be. An output-token cap keeps its meaning under the name the target has,
    and a target on Anthropic gets ``temperature`` or ``top_p``, not both, which
    current Claude models reject. A setting the target's field type refuses is
    left off with a warning rather than raised: a swap happens inside a kickoff.

    Args:
        model: The model string for the new instance, as ``LLM(model=...)``
            takes it.
        base: The instance whose configuration to carry. ``None`` (or anything
            that is not a :class:`BaseLLM`) contributes nothing.

    Returns:
        A new instance for ``model``; never ``base`` itself.
    """
    if not isinstance(base, BaseLLM):
        return LLM(model=model)

    # The declared endpoint decides where an unknown ``openai/`` model routes (a
    # custom OpenAI-compatible endpoint), so the same-provider question is asked
    # with it: a self-hosted model mapped to another model on the same endpoint
    # must keep that endpoint.
    route = LLM._resolve_route(model, _configured_settings(base, _ENDPOINT_NAMES))
    carried = _configured_settings(base, GENERATION_SETTINGS)
    if _same_provider(base, route):
        carried.update(_configured_settings(base, PROVIDER_SETTINGS))
    target = LLM._resolve_route(model, carried).native_class or LLM
    if type(base) is LLM and base.is_litellm:
        # The declared llm runs through LiteLLM — by the caller's choice or
        # because no native class knew its model; either way that is the
        # environment its callbacks and extra kwargs were written for.
        target = LLM
        carried["is_litellm"] = True
    accepted = {
        k: v
        for k, v in carried.items()
        if k in target.model_fields or k == "is_litellm"
    }
    if type(base) is not target:
        accepted.pop("additional_params", None)
    _carry_cap_under_the_targets_name(carried, accepted, target)
    if route.provider == "anthropic" and "temperature" in accepted:
        accepted.pop("top_p", None)
    return _build(model, accepted)


def _same_provider(base: BaseLLM, route: Any) -> bool:
    """Whether ``route`` lands where ``base``'s credentials and endpoint belong.

    A native class compares by class: an explicit ``provider=`` alias
    (``azure_openai``, ``claude``, ``google``) names the same class, and a
    user-defined :class:`BaseLLM` subclass is never a native one whatever
    ``provider`` string it defaulted to. A LiteLLM :class:`LLM` compares by the
    provider it resolved — to another LiteLLM route by the string, to a native
    route by the class that string names.
    """
    if route.native_class is not None:
        if type(base) is route.native_class:
            # One class serves every OpenAI-compatible provider (OpenRouter,
            # DeepSeek, Ollama, vLLM, …); each is its own vendor with its own
            # endpoint and key, so there the provider string decides.
            return not _serves_several_providers(route.native_class) or (
                route.provider == base.provider
            )
        return (
            type(base) is LLM
            and LLM._get_native_provider(base.provider or "") is route.native_class
        )
    return type(base) is LLM and route.provider == base.provider


def _serves_several_providers(native_class: type[BaseLLM]) -> bool:
    from crewai.llms.providers.openai_compatible.completion import (
        OpenAICompatibleCompletion,
    )

    return native_class is OpenAICompatibleCompletion


def _carry_cap_under_the_targets_name(
    carried: dict[str, Any], accepted: dict[str, Any], target: type[BaseLLM]
) -> None:
    """Give a configured output-token cap the name ``target`` has for it."""
    if any(name in accepted for name in _CAP_NAMES):
        return
    value = next((carried[name] for name in _CAP_NAMES if name in carried), None)
    if value is None:
        return
    for name in _CAP_NAMES:
        if name in target.model_fields:
            accepted[name] = value
            return


def _build(model: str, settings: dict[str, Any]) -> BaseLLM:
    """``LLM(model=..., **settings)``, without a setting the class refuses.

    A name the target has can still reject the value (LiteLLM's ``logprobs`` is
    an int, the OpenAI class's a bool). Such a setting is dropped, with a warning
    naming it, and the build retried; anything but a validation error propagates.
    """
    try:
        return LLM(model=model, **settings)
    except (ValidationError, ImportError) as exc:
        cause = exc if isinstance(exc, ValidationError) else exc.__cause__
        if not isinstance(cause, ValidationError):
            raise
        rejected = {
            str(err["loc"][0]) for err in cause.errors() if err.get("loc")
        } & set(settings)
        if not rejected:
            raise
        logger.warning(
            "llm_overlay: %s does not accept %s from the declared llm; built without",
            model,
            ", ".join(sorted(rejected)),
        )
        return _build(model, {k: v for k, v in settings.items() if k not in rejected})


def _configured_settings(base: BaseLLM, names: tuple[str, ...]) -> dict[str, Any]:
    """The values of ``names`` a caller configured on ``base``.

    Skips what is unset (``None``, an empty list or dict) and what ``base``'s
    class derived from its model; an Azure endpoint is carried as its resource
    root, without the deployment path the class appended for ``base``'s model.
    """
    settings: dict[str, Any] = {}
    for name in names:
        value = getattr(base, name, None)
        if value is None or (isinstance(value, (list, dict)) and not value):
            continue
        if name in _MODEL_DERIVED and _derived_from_model(base, name):
            continue
        if name == "endpoint" and isinstance(value, str):
            value = value.split(_AZURE_DEPLOYMENT_PATH)[0]
        settings[name] = value
    return settings


def _derived_from_model(base: BaseLLM, name: str) -> bool:
    """Whether ``base``'s class filled ``name`` from the model, not the caller.

    Only a class that gives the field a default of its own can have derived it
    (Anthropic's ``max_tokens``; elsewhere the default is ``None`` and the value
    is the caller's, whatever the class's ``to_config_dict`` chooses to emit).
    For such a class, ``to_config_dict`` is its own account of what was
    configured: Anthropic leaves ``max_tokens`` out of it when the value is the
    cap it derived for the model at construction — so a caller's value that
    happens to equal that cap counts as derived, and the new model derives its
    own; the conservative side.
    """
    field = type(base).model_fields.get(name)
    if field is None or field.default is None:
        return False
    if type(base).to_config_dict is BaseLLM.to_config_dict:
        return False
    return name not in base.to_config_dict()


UNACCEPTED_ATTRIBUTES: Final[list[str]] = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
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

    unaccepted_attributes = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]
    set_provider = model_name.partition("/")[0] if "/" in model_name else "openai"

    if set_provider in ENV_VARS:
        env_vars_for_provider = ENV_VARS[set_provider]
        if isinstance(env_vars_for_provider, (list, tuple)):
            for env_var in env_vars_for_provider:
                key_name = env_var.get("key_name")
                if key_name and key_name not in unaccepted_attributes:
                    env_value = os.environ.get(key_name)
                    if env_value:
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
        logger.error(
            f"Error instantiating LLM from environment/fallback: {type(e).__name__}: {e}"
        )
        raise e


def _normalize_key_name(key_name: str) -> str:
    """Maps environment variable names to recognized litellm parameter keys.

    Args:
        key_name: The environment variable name to normalize.
    """
    for pattern in LITELLM_PARAMS:
        if pattern in key_name:
            return pattern
    return key_name
