from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeGuard, TypeVar

from pydantic import BaseModel

from crewai.utilities.logger_utils import suppress_warnings


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.llm import LLM
    from crewai.llms.base_llm import BaseLLM
    from crewai.utilities.types import LLMMessage


T = TypeVar("T", bound=BaseModel)


def _is_valid_llm(llm: Any) -> TypeGuard[str | LLM | BaseLLM]:
    """Type guard to ensure LLM is valid and not None.

    Args:
        llm: The LLM to validate

    Returns:
        True if LLM is valid (string or has model attribute), False otherwise
    """
    return llm is not None and (isinstance(llm, str) or hasattr(llm, "model"))


class InternalInstructor(Generic[T]):
    """Class that wraps an agent LLM with instructor for structured output generation.

    Attributes:
        content: The content to be processed
        model: The Pydantic model class for the response
        agent: The agent with LLM
        llm: The LLM instance or model name
    """

    def __init__(
        self,
        content: str,
        model: type[T],
        agent: Agent | None = None,
        llm: LLM | BaseLLM | str | None = None,
    ) -> None:
        """Initialize InternalInstructor.

        Args:
            content: The content to be processed
            model: The Pydantic model class for the response
            agent: The agent with LLM
            llm: The LLM instance or model name
        """
        self.content = content
        self.agent = agent
        self.model = model
        self.llm = llm or (agent.function_calling_llm or agent.llm if agent else None)

        with suppress_warnings():
            import instructor  # type: ignore[import-untyped]

            if (
                self.llm is not None
                and hasattr(self.llm, "is_litellm")
                and self.llm.is_litellm
            ):
                from litellm import completion

                self._client = instructor.from_litellm(completion)
            else:
                self._client = self._create_instructor_client()

    def _get_llm_client_kwargs(self) -> dict[str, Any]:
        """Extract base_url and api_key from the LLM so they can be forwarded
        to the instructor client.

        Returns:
            Dict with ``base_url`` and/or ``api_key`` when present on the LLM.
        """
        kwargs: dict[str, Any] = {}
        if isinstance(self.llm, str):
            return kwargs
        for attr in ("base_url", "api_base", "api_key"):
            value = getattr(self.llm, attr, None)
            if value is not None:
                # Normalize api_base → base_url for the OpenAI client.
                # First writer wins: base_url takes precedence over api_base.
                key = "base_url" if attr == "api_base" else attr
                if key not in kwargs:
                    kwargs[key] = value
        return kwargs

    def _create_instructor_client(self) -> Any:
        """Create instructor client configured for the LLM provider.

        When the LLM carries a custom ``base_url`` (e.g. self-hosted vLLM,
        Ollama, or any OpenAI-compatible endpoint), we construct an explicit
        provider client so the URL is not silently discarded.

        Returns:
            Instructor client configured for the LLM provider

        Raises:
            ValueError: If the provider is not supported
        """
        import instructor

        if isinstance(self.llm, str):
            model_string = self.llm
        elif self.llm is not None and hasattr(self.llm, "model"):
            model_string = self.llm.model
        else:
            raise ValueError("LLM must be a string or have a model attribute")

        if isinstance(self.llm, str):
            provider = self._extract_provider()
        elif self.llm is not None and hasattr(self.llm, "provider"):
            provider = self.llm.provider
        else:
            provider = "openai"

        extra_kwargs = self._get_llm_client_kwargs()

        # If the LLM has a custom base_url we must construct the provider
        # client explicitly — instructor.from_provider() does not forward
        # base_url to the underlying SDK client constructor.
        if "base_url" in extra_kwargs:
            return self._create_instructor_client_with_base_url(
                provider, model_string, extra_kwargs
            )

        return instructor.from_provider(f"{provider}/{model_string}")

    def _create_instructor_client_with_base_url(
        self, provider: str, model_string: str, kwargs: dict[str, Any]
    ) -> Any:
        """Create an instructor client using an explicit SDK client so that
        ``base_url`` is forwarded correctly.

        Args:
            provider: The provider name (e.g. ``"openai"``, ``"anthropic"``).
            model_string: The model identifier.
            kwargs: Dict containing ``base_url`` and optionally ``api_key``.

        Returns:
            An Instructor client wrapping a provider-specific SDK client.
        """
        import instructor

        base_url = kwargs.get("base_url")
        api_key = kwargs.get("api_key")

        if provider in ("azure", "azure_openai"):
            from openai import AzureOpenAI

            client_kwargs: dict[str, Any] = {}
            if base_url is not None:
                client_kwargs["azure_endpoint"] = base_url
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            api_version = getattr(self.llm, "api_version", None)
            if api_version is not None:
                client_kwargs["api_version"] = api_version
            return instructor.from_openai(AzureOpenAI(**client_kwargs))

        if provider == "openai":
            from openai import OpenAI

            client_kwargs = {}
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            return instructor.from_openai(OpenAI(**client_kwargs))

        if provider == "anthropic":
            from anthropic import Anthropic

            client_kwargs = {}
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            return instructor.from_anthropic(Anthropic(**client_kwargs))

        # For other providers, fall back to from_provider. base_url may not
        # be supported, but we forward api_key if available.
        fp_kwargs: dict[str, Any] = {}
        if api_key is not None:
            fp_kwargs["api_key"] = api_key
        return instructor.from_provider(f"{provider}/{model_string}", **fp_kwargs)

    def _extract_provider(self) -> str:
        """Extract provider from LLM model name.

        Returns:
            Provider name (e.g., 'openai', 'anthropic', etc.)
        """
        if self.llm is not None and hasattr(self.llm, "provider") and self.llm.provider:
            return self.llm.provider

        if isinstance(self.llm, str):
            return self.llm.partition("/")[0] or "openai"
        if self.llm is not None and hasattr(self.llm, "model"):
            return self.llm.model.partition("/")[0] or "openai"
        return "openai"

    def to_json(self) -> str:
        """Convert the structured output to JSON format.

        Returns:
            JSON string representation of the structured output
        """
        pydantic_model = self.to_pydantic()
        return pydantic_model.model_dump_json(indent=2)

    def to_pydantic(self) -> T:
        """Generate structured output using the specified Pydantic model.

        Returns:
            Instance of the specified Pydantic model with structured data

        Raises:
            ValueError: If LLM is not provided or invalid
        """
        messages: list[LLMMessage] = [{"role": "user", "content": self.content}]

        if not _is_valid_llm(self.llm):
            raise ValueError(
                "LLM must be provided and have a model attribute or be a string"
            )

        if isinstance(self.llm, str):
            model_name = self.llm
        else:
            model_name = self.llm.model

        return self._client.chat.completions.create(  # type: ignore[no-any-return]
            model=model_name, response_model=self.model, messages=messages
        )
