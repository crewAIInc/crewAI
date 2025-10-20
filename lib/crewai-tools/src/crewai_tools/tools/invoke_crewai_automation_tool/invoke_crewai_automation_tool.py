import time
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, create_model
import requests


class InvokeCrewAIAutomationInput(BaseModel):
    """Input schema for InvokeCrewAIAutomationTool."""

    prompt: str = Field(..., description="The prompt or query to send to the crew")


class InvokeCrewAIAutomationTool(BaseTool):
    """A CrewAI tool for invoking external crew/flows APIs.

    This tool provides CrewAI Platform API integration with external crew services, supporting:
    - Dynamic input schema configuration
    - Automatic polling for task completion
    - Bearer token authentication
    - Comprehensive error handling

    Example:
        Basic usage:
        >>> tool = InvokeCrewAIAutomationTool(
        ...     crew_api_url="https://api.example.com",
        ...     crew_bearer_token="your_token",
        ...     crew_name="My Crew",
        ...     crew_description="Description of what the crew does",
        ... )

        With custom inputs:
        >>> custom_inputs = {
        ...     "param1": Field(..., description="Description of param1"),
        ...     "param2": Field(
        ...         default="default_value", description="Description of param2"
        ...     ),
        ... }
        >>> tool = InvokeCrewAIAutomationTool(
        ...     crew_api_url="https://api.example.com",
        ...     crew_bearer_token="your_token",
        ...     crew_name="My Crew",
        ...     crew_description="Description of what the crew does",
        ...     crew_inputs=custom_inputs,
        ... )

    Example:
        >>> tools = [
        ...     InvokeCrewAIAutomationTool(
        ...         crew_api_url="https://canary-crew-[...].crewai.com",
        ...         crew_bearer_token="[Your token: abcdef012345]",
        ...         crew_name="State of AI Report",
        ...         crew_description="Retrieves a report on state of AI for a given year.",
        ...         crew_inputs={
        ...             "year": Field(
        ...                 ..., description="Year to retrieve the report for (integer)"
        ...             )
        ...         },
        ...     )
        ... ]
    """

    name: str = "invoke_amp_automation"
    description: str = "Invokes an CrewAI Platform Automation using API"
    args_schema: type[BaseModel] = InvokeCrewAIAutomationInput

    crew_api_url: str
    crew_bearer_token: str
    max_polling_time: int = 10 * 60  # 10 minutes

    def __init__(
        self,
        crew_api_url: str,
        crew_bearer_token: str,
        crew_name: str,
        crew_description: str,
        max_polling_time: int = 10 * 60,
        crew_inputs: dict[str, Any] | None = None,
    ):
        """Initialize the InvokeCrewAIAutomationTool.

        Args:
            crew_api_url: Base URL of the crew API service
            crew_bearer_token: Bearer token for API authentication
            crew_name: Name of the crew to invoke
            crew_description: Description of the crew to invoke
            max_polling_time: Maximum time in seconds to wait for task completion (default: 600 seconds = 10 minutes)
            crew_inputs: Optional dictionary defining custom input schema fields
        """
        # Create dynamic args_schema if custom inputs provided
        if crew_inputs:
            # Start with the base prompt field
            fields = {}

            # Add custom fields
            for field_name, field_def in crew_inputs.items():
                if isinstance(field_def, tuple):
                    fields[field_name] = field_def
                else:
                    # Assume it's a Field object, extract type from annotation if available
                    fields[field_name] = (str, field_def)

            # Create dynamic model
            args_schema = create_model("DynamicInvokeCrewAIAutomationInput", **fields)  # type: ignore[call-overload]
        else:
            args_schema = InvokeCrewAIAutomationInput

        # Initialize the parent class with proper field values
        super().__init__(
            name=crew_name,
            description=crew_description,
            args_schema=args_schema,
            crew_api_url=crew_api_url,
            crew_bearer_token=crew_bearer_token,
            max_polling_time=max_polling_time,
        )

    def _kickoff_crew(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Start a new crew task.

        Args:
            inputs: Dictionary containing the query and other input parameters

        Returns:
            Dictionary containing the crew task response. The response will contain the crew id which needs to be returned to check the status of the crew.
        """
        response = requests.post(
            f"{self.crew_api_url}/kickoff",
            headers={
                "Authorization": f"Bearer {self.crew_bearer_token}",
                "Content-Type": "application/json",
            },
            json={"inputs": inputs},
            timeout=30,
        )
        return response.json()

    def _get_crew_status(self, crew_id: str) -> dict[str, Any]:
        """Get the status of a crew task.

        Args:
            crew_id: The ID of the crew task to check

        Returns:
            Dictionary containing the crew task status
        """
        response = requests.get(
            f"{self.crew_api_url}/status/{crew_id}",
            headers={
                "Authorization": f"Bearer {self.crew_bearer_token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        return response.json()

    def _run(self, **kwargs) -> str:
        """Execute the crew invocation tool."""
        if kwargs is None:
            kwargs = {}

        # Start the crew
        response = self._kickoff_crew(inputs=kwargs)
        kickoff_id: str | None = response.get("kickoff_id")

        if kickoff_id is None:
            return f"Error: Failed to kickoff crew. Response: {response}"

        # Poll for completion
        for i in range(self.max_polling_time):
            try:
                status_response = self._get_crew_status(crew_id=kickoff_id)
                if status_response.get("state", "").lower() == "success":
                    return status_response.get("result", "No result returned")
                if status_response.get("state", "").lower() == "failed":
                    return f"Error: Crew task failed. Response: {status_response}"
            except Exception as e:
                if i == self.max_polling_time - 1:  # Last attempt
                    return f"Error: Failed to get crew status after {self.max_polling_time} attempts. Last error: {e}"

            time.sleep(1)

        return f"Error: Crew did not complete within {self.max_polling_time} seconds"
