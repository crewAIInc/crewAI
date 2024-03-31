from typing import Any, Optional, Type

import instructor
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class Instructor(BaseModel):
    """Class that wraps an agent llm with instructor."""

    _client: Any = PrivateAttr()
    content: str = Field(description="Content to be sent to the instructor.")
    agent: Optional[Any] = Field(
        description="The agent that needs to use instructor.", default=None
    )
    llm: Optional[Any] = Field(
        description="The agent that needs to use instructor.", default=None
    )
    instructions: Optional[str] = Field(
        description="Instructions to be sent to the instructor.",
        default=None,
    )
    model: Type[BaseModel] = Field(
        description="Pydantic model to be used to create an output."
    )

    @model_validator(mode="after")
    def set_instructor(self):
        """Set instructor."""
        if self.agent and not self.llm:
            self.llm = self.agent.function_calling_llm or self.agent.llm

        self._client = instructor.patch(
            self.llm.client._client,
            mode=instructor.Mode.TOOLS,
        )
        return self

    def to_json(self):
        model = self.to_pydantic()
        return model.model_dump_json(indent=2)

    def to_pydantic(self):
        messages = [{"role": "user", "content": self.content}]
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})

        model = self._client.chat.completions.create(
            model=self.llm.model_name, response_model=self.model, messages=messages
        )
        return model
