import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from crewai.tasks.output_format import OutputFormat


class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    name: Optional[str] = Field(description="Name of the task", default=None)
    expected_output: Optional[str] = Field(
        description="Expected output of the task", default=None
    )
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    raw: str = Field(description="Raw output of the task", default="")
    pydantic: Optional[BaseModel] = Field(
        description="Pydantic output of task", default=None
    )
    json_dict: Optional[Dict[str, Any]] = Field(
        description="JSON dictionary of task", default=None
    )
    agent: str = Field(description="Agent that executed the task")
    output_format: OutputFormat = Field(
        description="Output format of the task", default=OutputFormat.RAW
    )
    completion_metadata: Optional[Dict[str, Any]] = Field(
        description="Full completion metadata including generations and logprobs", default=None
    )

    @model_validator(mode="after")
    def set_summary(self):
        """Set the summary field based on the description."""
        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self

    @property
    def json(self) -> Optional[str]:
        if self.output_format != OutputFormat.JSON:
            raise ValueError(
                """
                Invalid output format requested.
                If you would like to access the JSON output,
                please make sure to set the output_json property for the task
                """
            )

        return json.dumps(self.json_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert json_output and pydantic_output to a dictionary."""
        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        elif self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict

    def get_generations(self) -> Optional[List[str]]:
        """Get all generations from completion metadata."""
        if not self.completion_metadata or "choices" not in self.completion_metadata:
            return None
        
        generations = []
        for choice in self.completion_metadata["choices"]:
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                generations.append(choice.message.content or "")
            elif isinstance(choice, dict) and "message" in choice:
                generations.append(choice["message"].get("content", ""))
        
        return generations if generations else None

    def get_logprobs(self) -> Optional[List[Dict[str, Any]]]:
        """Get log probabilities from completion metadata."""
        if not self.completion_metadata or "choices" not in self.completion_metadata:
            return None
        
        logprobs_list = []
        for choice in self.completion_metadata["choices"]:
            if hasattr(choice, "logprobs") and choice.logprobs:
                logprobs_list.append(choice.logprobs)
            elif isinstance(choice, dict) and "logprobs" in choice:
                logprobs_list.append(choice["logprobs"])
        
        return logprobs_list if logprobs_list else None

    def get_usage_metrics(self) -> Optional[Dict[str, Any]]:
        """Get token usage metrics from completion metadata."""
        if not self.completion_metadata:
            return None
        return self.completion_metadata.get("usage")

    def __str__(self) -> str:
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        return self.raw
