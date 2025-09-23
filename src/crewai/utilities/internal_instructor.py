from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeGuard, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.llm import LLM
    from crewai.llms.base_llm import BaseLLM

from crewai.utilities.logger_utils import suppress_warnings
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
            import instructor
            from litellm import completion

            self._client = instructor.from_litellm(completion)

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

        return self._client.chat.completions.create(
            model=model_name, response_model=self.model, messages=messages
        )
