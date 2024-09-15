from typing import Any, Optional, Type

import instructor
from litellm import completion


class InternalInstructor:
    """Class that wraps an agent llm with instructor."""

    def __init__(
        self,
        content: str,
        model: Type,
        agent: Optional[Any] = None,
        llm: Optional[str] = None,
        instructions: Optional[str] = None,
    ):
        self.content = content
        self.agent = agent
        self.llm = llm
        self.instructions = instructions
        self.model = model
        self._client = None
        self.set_instructor()

    def set_instructor(self):
        """Set instructor."""
        if self.agent and not self.llm:
            self.llm = self.agent.function_calling_llm or self.agent.llm

        self._client = instructor.from_litellm(
            completion,
            mode=instructor.Mode.TOOLS,
        )

    def to_json(self):
        model = self.to_pydantic()
        return model.model_dump_json(indent=2)

    def to_pydantic(self):
        messages = [{"role": "user", "content": self.content}]
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        model = self._client.chat.completions.create(
            model=self.llm, response_model=self.model, messages=messages
        )
        return model
