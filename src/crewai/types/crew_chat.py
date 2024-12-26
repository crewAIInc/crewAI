from typing import List, Optional

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

    name: str
    description: Optional[str] = None


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

    crew_name: Optional[str] = Field(default="Crew")
    crew_description: Optional[str] = None
    inputs: List[ChatInputField] = Field(default_factory=list)
