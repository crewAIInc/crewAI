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
        self._litellm_api_base: str | None = None
        self._instructor_model_name: str = ""

        with suppress_warnings():
            import instructor  # type: ignore[import-untyped]

            if (
                self.llm is not None
                and hasattr(self.llm, "is_litellm")
                and self.llm.is_litellm
            ):
                from litellm import completion

                self._client = instructor.from_litellm(completion)
                _base = getattr(self.llm, "base_url", None) or getattr(
                    self.llm, "api_base", None
                )
                if _base:
                    self._litellm_api_base = str(_base)
                # Litellm uses the provider-prefixed model string for routing.
                if hasattr(self.llm, "model"):
                    self._instructor_model_name = str(self.llm.model)
                else:
                    raise ValueError(
                        "LLM must have a model attribute when used with litellm"
                    )
            else:
                self._client = self._create_instructor_client()
                # _instructor_model_name is set inside _create_instructor_client()

    def _create_instructor_client(self) -> Any:
        """Create instructor client configured for the LLM provider.

        When the LLM carries a custom ``base_url`` (e.g. self-hosted vLLM,
        Ollama, Anthropic proxy, or Azure OpenAI), constructs an explicit
        provider-specific SDK client so the URL is not silently discarded by
        ``instructor.from_provider()``.

        Provider routing when ``base_url`` is set:
        - ``anthropic``                → ``Anthropic(base_url=...)`` via ``instructor.from_anthropic()``
        - ``azure`` / ``azure_openai`` → ``AzureOpenAI(azure_endpoint=...)`` via ``instructor.from_openai()``
        - everything else              → ``OpenAI(base_url=...)`` via ``instructor.from_openai()``

        ``base_url`` takes precedence over ``api_base`` when both are present.

        Returns:
            Instructor client configured for the LLM provider

        Raises:
            ValueError: If the LLM is not valid
        """
        import instructor

        # Resolve model_string and provider in a single dispatch to avoid
        # redundant isinstance checks.  model_string strips any provider prefix
        # (e.g. "anthropic/claude-3-5-sonnet" → "claude-3-5-sonnet") so that
        # direct SDK clients receive a bare model id while from_provider can
        # reconstruct the full routing string.
        if isinstance(self.llm, str):
            model_string = self.llm.partition("/")[2] or self.llm
            provider = self._extract_provider()
        elif self.llm is not None and hasattr(self.llm, "model"):
            model_string = self.llm.model.partition("/")[2] or self.llm.model
            provider = getattr(self.llm, "provider", None) or "openai"
        else:
            raise ValueError("LLM must be a string or have a model attribute")

        # Normalise api_base → base_url; base_url takes precedence when both are set.
        # Empty strings are treated the same as absent (normalised to None).
        base_url: str | None = getattr(self.llm, "base_url", None) or None
        if base_url is None:
            base_url = getattr(self.llm, "api_base", None) or None
        api_key: str | None = getattr(self.llm, "api_key", None) or None

        # Casefold for case-insensitive matching (e.g. "Anthropic" == "anthropic").
        # from_provider still receives the original casing preserved by the LLM.
        provider_lower = provider.casefold() if provider else "openai"

        if base_url:
            if provider_lower == "anthropic":
                try:
                    from anthropic import Anthropic
                except ImportError as exc:
                    raise ImportError(
                        "The 'anthropic' package is required for Anthropic provider support. "
                        "Install it with: pip install anthropic"
                    ) from exc

                self._instructor_model_name = model_string
                client_kwargs: dict[str, Any] = {"base_url": base_url}
                if api_key is not None:
                    client_kwargs["api_key"] = api_key
                return instructor.from_anthropic(Anthropic(**client_kwargs))

            if provider_lower in ("azure", "azure_openai"):
                from openai import AzureOpenAI

                self._instructor_model_name = model_string
                client_kwargs = {"azure_endpoint": base_url}
                if api_key is not None:
                    client_kwargs["api_key"] = api_key
                api_version: str | None = getattr(self.llm, "api_version", None) or None
                if api_version is not None:
                    client_kwargs["api_version"] = api_version
                azure_deployment: str | None = getattr(self.llm, "azure_deployment", None) or None
                if azure_deployment is not None:
                    client_kwargs["azure_deployment"] = azure_deployment
                return instructor.from_openai(AzureOpenAI(**client_kwargs))

            # OpenAI and all other OpenAI-compatible providers (vLLM, Ollama, Groq, …)
            # Keyless local servers (Ollama, vLLM) still require a non-empty api_key
            # for the openai SDK constructor; use a placeholder when none is provided.
            from openai import OpenAI

            self._instructor_model_name = model_string
            client_kwargs = {"base_url": base_url, "api_key": api_key or "not-needed"}
            return instructor.from_openai(OpenAI(**client_kwargs))

        # from_provider requires the full "provider/model" routing string.
        self._instructor_model_name = f"{provider}/{model_string}"
        return instructor.from_provider(f"{provider}/{model_string}")

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

        extra: dict[str, Any] = {}
        if self._litellm_api_base:
            extra["api_base"] = self._litellm_api_base

        return self._client.chat.completions.create(  # type: ignore[no-any-return]
            model=self._instructor_model_name,
            response_model=self.model,
            messages=messages,
            **extra,
        )
