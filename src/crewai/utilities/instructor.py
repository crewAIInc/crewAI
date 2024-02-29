from typing import Any, Optional, Type

import instructor
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class Instructor(BaseModel):
    """This class is designed to wrap a language model (llm) with an instructor. 
    The instructor guides the language model in generating responses based on the provided instructions."""

    _client: Any = PrivateAttr()  # Private attribute to store the client that communicates with the language model
    content: str = Field(description="The content that will be sent to the language model for processing.")
    agent: Optional[Any] = Field(
        description="The agent that will use the instructor. If not provided, the llm will be used directly.", 
        default=None
    )
    llm: Optional[Any] = Field(
        description="The language model that will be guided by the instructor. If not provided, it will be obtained from the agent.", 
        default=None
    )
    instructions: Optional[str] = Field(
        description="The instructions that will be sent to the language model to guide its response generation.",
        default=None,
    )
    model: Type[BaseModel] = Field(
        description="The Pydantic model that will be used to structure the output from the language model."
    )

    @model_validator(mode="after")
    def set_instructor(self):
        """This method sets up the instructor after the model validation.

        If an agent is provided but an llm is not, the llm is obtained from the agent. 
        The client is then patched with the instructor in the 'TOOLS' mode.
        """
        if self.agent and not self.llm:
            self.llm = self.agent.function_calling_llm or self.agent.llm

        self._client = instructor.patch(
            self.llm.client._client,
            mode=instructor.Mode.TOOLS,
        )
        return self

    def to_json(self):
        """This method converts the output from the language model into a JSON string."""
        model = self.to_pydantic()
        return model.model_dump_json(indent=2)

    def to_pydantic(self):
        """This method generates a response from the language model and structures it into a Pydantic model.

        The content and instructions (if provided) are sent to the language model. The generated response is then
        structured into the specified Pydantic model.
        """
        messages = [{"role": "user", "content": self.content}]
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})

        model = self._client.chat.completions.create(
            model=self.llm.model_name, response_model=self.model, messages=messages
        )
        return model
