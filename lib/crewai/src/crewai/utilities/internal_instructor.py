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
        self._async_client: Any = None

        with suppress_warnings():
            import instructor

            if (
                self.llm is not None
                and hasattr(self.llm, "is_litellm")
                and self.llm.is_litellm
            ):
                from litellm import completion

                self._client = instructor.from_litellm(completion)
            else:
                self._client = self._create_instructor_client()

    def _build_provider_model_string(self) -> str:
        """Build a ``provider/model`` string for instructor.

        Returns:
            A string like ``"openai/gpt-4o"``

        Raises:
            ValueError: If the LLM has no model attribute
        """
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

        return f"{provider}/{model_string}"

    def _create_instructor_client(self) -> Any:
        """Create instructor client using the modern from_provider pattern.

        Returns:
            Instructor client configured for the LLM provider
        """
        import instructor

        return instructor.from_provider(self._build_provider_model_string())

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

    async def ato_json(self) -> str:
        """Async convert the structured output to JSON format.

        Returns:
            JSON string representation of the structured output
        """
        pydantic_model = await self.ato_pydantic()
        return pydantic_model.model_dump_json(indent=2)

    async def ato_pydantic(self) -> T:
        """Async generate structured output using the specified Pydantic model.

        Uses an async instructor client so the event loop is never blocked.

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

        async_client = self._get_async_client()
        return await async_client.chat.completions.create(  # type: ignore[no-any-return]
            model=model_name, response_model=self.model, messages=messages
        )

    def _get_async_client(self) -> Any:
        """Create or return a cached async instructor client.

        Returns:
            Async instructor client configured for the LLM provider
        """
        if self._async_client is not None:
            return self._async_client

        with suppress_warnings():
            import instructor

            if (
                self.llm is not None
                and hasattr(self.llm, "is_litellm")
                and self.llm.is_litellm
            ):
                from litellm import acompletion

                self._async_client = instructor.from_litellm(acompletion)
            else:
                self._async_client = self._create_async_instructor_client()

        return self._async_client

    def _create_async_instructor_client(self) -> Any:
        """Create an async instructor client using the from_provider pattern.

        Returns:
            Async instructor client configured for the LLM provider
        """
        import instructor

        return instructor.from_provider(
            self._build_provider_model_string(), async_mode="async"
        )
