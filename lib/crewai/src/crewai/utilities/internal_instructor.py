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

        if isinstance(self.llm, str):
            # Strip provider prefix so "anthropic/claude-3-5-sonnet" → "claude-3-5-sonnet"
            model_string = self.llm.partition("/")[2] or self.llm
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

        # Normalise api_base → base_url; base_url takes precedence when both are set.
        base_url: str | None = getattr(self.llm, "base_url", None)
        if base_url is None:
            base_url = getattr(self.llm, "api_base", None)
        api_key: str | None = getattr(self.llm, "api_key", None)

        if base_url:
            if provider == "anthropic":
                from anthropic import Anthropic

                client_kwargs: dict[str, Any] = {"base_url": base_url}
                if api_key is not None:
                    client_kwargs["api_key"] = api_key
                return instructor.from_anthropic(Anthropic(**client_kwargs))

            if provider in ("azure", "azure_openai"):
                from openai import AzureOpenAI

                client_kwargs = {"azure_endpoint": base_url}
                if api_key is not None:
                    client_kwargs["api_key"] = api_key
                api_version: str | None = getattr(self.llm, "api_version", None)
                if api_version is not None:
                    client_kwargs["api_version"] = api_version
                return instructor.from_openai(AzureOpenAI(**client_kwargs))

            # OpenAI and all other OpenAI-compatible providers (vLLM, Ollama, Groq, …)
            from openai import OpenAI

            client_kwargs = {"base_url": base_url}
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            return instructor.from_openai(OpenAI(**client_kwargs))

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

        if isinstance(self.llm, str):
            model_name = self.llm
        else:
            model_name = self.llm.model

        return self._client.chat.completions.create(  # type: ignore[no-any-return]
            model=model_name, response_model=self.model, messages=messages
        )
