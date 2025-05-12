import warnings
from typing import Any


class InternalInstructor:
    """Class that wraps an agent llm with instructor."""

    def __init__(
        self,
        content: str,
        model: type,
        agent: Any | None = None,
        llm: str | None = None,
    ) -> None:
        self.content = content
        self.agent = agent
        self.llm = llm
        self.model = model
        self._client = None
        self.set_instructor()

    def set_instructor(self) -> None:
        """Set instructor."""
        if self.agent and not self.llm:
            self.llm = self.agent.function_calling_llm or self.agent.llm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            import instructor
            from litellm import completion

            self._client = instructor.from_litellm(completion)

    def to_json(self):
        model = self.to_pydantic()
        return model.model_dump_json(indent=2)

    def to_pydantic(self):
        messages = [{"role": "user", "content": self.content}]
        return self._client.chat.completions.create(
            model=self.llm.model, response_model=self.model, messages=messages,
        )
