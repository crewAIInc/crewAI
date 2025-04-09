from typing import List

from pydantic import BaseModel, Field


class ChatInputField(BaseModel):
    """
    Represents a single required input for the crew, with a name and short description.
    Example:
        {
            "name": "topic",
            "description": "The topic to focus on for the conversation"
        }
    """

    name: str = Field(..., description="The name of the input field")
    description: str = Field(..., description="A short description of the input field")


class ChatInputs(BaseModel):
    """
    Holds a high-level crew_description plus a list of ChatInputFields.
    Example:
        {
            "crew_name": "topic-based-qa",
            "crew_description": "Use this crew for topic-based Q&A",
            "inputs": [
                {"name": "topic", "description": "The topic to focus on"},
                {"name": "username", "description": "Name of the user"},
            ]
        }
    """

    crew_name: str = Field(..., description="The name of the crew")
    crew_description: str = Field(
        ..., description="A description of the crew's purpose"
    )
    inputs: List[ChatInputField] = Field(
        default_factory=list, description="A list of input fields for the crew"
    )
