from abc import ABC, abstractmethod


class BaseConverterAdapter(ABC):
    def __init__(self, agent_adapter):
        self.agent_adapter = agent_adapter

    @abstractmethod
    def configure_structured_output(self, task) -> None:
        """Configure agents to return structured output.
        Must support json and pydantic output.
        """
        pass

    @abstractmethod
    def enhance_system_prompt(self, base_prompt: str) -> str:
        """Enhance the system prompt with structured output instructions."""
        pass

    @abstractmethod
    def post_process_result(self, result: str) -> str:
        """Post-process the result to ensure it matches the expected format."""
        # Transform the string result to the expected format.
        pass
