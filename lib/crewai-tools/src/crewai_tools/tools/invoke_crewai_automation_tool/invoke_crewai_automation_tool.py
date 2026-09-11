import os
import time
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, create_model
import requests


# Generic fallbacks used when the tool is instantiated without a crew_name /
# crew_description, e.g. by a runtime that resolves tools by class reference
# and instantiates them with no arguments (see the "Zero-argument
# instantiation" note in the class docstring below).
DEFAULT_TOOL_NAME = "invoke_amp_automation"
DEFAULT_TOOL_DESCRIPTION = "Invokes an CrewAI Platform Automation using API"


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

        Configuring the API url and bearer token via environment variables, instead of
        passing them as constructor arguments:
        >>> import os
        >>> os.environ["CREWAI_API_URL"] = "https://canary-crew-[...].crewai.com"
        >>> os.environ["CREWAI_BEARER_TOKEN"] = "[Your token: abcdef012345]"
        >>> tool = InvokeCrewAIAutomationTool(
        ...     crew_name="State of AI Report",
        ...     crew_description="Retrieves a report on state of AI for a given year.",
        ... )

    Zero-argument instantiation:
        `crew_api_url`/`crew_bearer_token` fall back to the `CREWAI_API_URL` /
        `CREWAI_BEARER_TOKEN` environment variables, and `crew_name` /
        `crew_description` fall back to a generic name/description, so
        `InvokeCrewAIAutomationTool()` never raises at construction time. This
        matters because some runtimes (e.g. the CrewAI AMP Studio "Invoke Amp
        Automation" internal tool) resolve tools by class reference and instantiate
        them with no arguments. If the tool is still unconfigured (no explicit
        arguments and no environment variables) when it is actually invoked, it
        raises a clear `ValueError` instead of attempting the HTTP request.
    """

    name: str = DEFAULT_TOOL_NAME
    description: str = DEFAULT_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = InvokeCrewAIAutomationInput

    crew_api_url: str | None = None
    crew_bearer_token: str | None = None
    max_polling_time: int = 10 * 60
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="CREWAI_API_URL",
                description="Base URL of the crew/flow API to invoke. Alternative to passing crew_api_url.",
                required=True,
            ),
            EnvVar(
                name="CREWAI_BEARER_TOKEN",
                description="Bearer token used to authenticate against crew_api_url. Alternative to passing crew_bearer_token.",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        crew_api_url: str | None = None,
        crew_bearer_token: str | None = None,
        crew_name: str = DEFAULT_TOOL_NAME,
        crew_description: str = DEFAULT_TOOL_DESCRIPTION,
        max_polling_time: int = 10 * 60,
        crew_inputs: dict[str, Any] | None = None,
    ):
        """Initialize the InvokeCrewAIAutomationTool.

        Args:
            crew_api_url: Base URL of the crew API service. If omitted, falls back to
                the CREWAI_API_URL environment variable.
            crew_bearer_token: Bearer token for API authentication. If omitted, falls
                back to the CREWAI_BEARER_TOKEN environment variable.
            crew_name: Name of the crew to invoke. Defaults to a generic tool name so
                the tool can be instantiated without arguments; set it explicitly so
                the LLM sees a meaningful tool name in its tool list.
            crew_description: Description of the crew to invoke. Defaults to a generic
                description for the same reason as crew_name.
            max_polling_time: Maximum time in seconds to wait for task completion (default: 600 seconds = 10 minutes)
            crew_inputs: Optional dictionary defining custom input schema fields
        """
        if crew_inputs:
            fields = {}

            for field_name, field_def in crew_inputs.items():
                if isinstance(field_def, tuple):
                    fields[field_name] = field_def
                else:
                    # Assume it's a Field object, extract type from annotation if available
                    fields[field_name] = (str, field_def)

            args_schema = create_model("DynamicInvokeCrewAIAutomationInput", **fields)  # type: ignore[call-overload]
        else:
            args_schema = InvokeCrewAIAutomationInput

        # Explicit constructor arguments win over the environment variables.
        resolved_api_url = crew_api_url or os.getenv("CREWAI_API_URL")
        resolved_bearer_token = crew_bearer_token or os.getenv("CREWAI_BEARER_TOKEN")

        super().__init__(
            name=crew_name or DEFAULT_TOOL_NAME,
            description=crew_description or DEFAULT_TOOL_DESCRIPTION,
            args_schema=args_schema,
            crew_api_url=resolved_api_url,
            crew_bearer_token=resolved_bearer_token,
            max_polling_time=max_polling_time,
        )

    def _ensure_configured(self) -> None:
        """Raise a clear, actionable error if the tool has no API url/token.

        crew_api_url and crew_bearer_token are optional at construction time (they
        may be resolved later from the environment, or simply not set yet when the
        tool is instantiated with no arguments). This is checked once, right before
        the tool is actually used, so a missing configuration produces an explicit
        error message instead of a confusing failure deep inside `requests` (e.g. an
        "Invalid URL 'None/kickoff'" error) or a bare 401 from the API.
        """
        missing = [
            env_name
            for value, env_name in (
                (self.crew_api_url, "CREWAI_API_URL"),
                (self.crew_bearer_token, "CREWAI_BEARER_TOKEN"),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "InvokeCrewAIAutomationTool is not configured: missing "
                f"{' and '.join(missing)}. Pass crew_api_url/crew_bearer_token "
                "explicitly when creating the tool, or set the "
                f"{' and '.join(missing)} environment variable(s)."
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
        result: dict[str, Any] = response.json()
        return result

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
        result: dict[str, Any] = response.json()
        return result

    def _run(self, **kwargs: Any) -> str:
        """Execute the crew invocation tool."""
        self._ensure_configured()

        if kwargs is None:
            kwargs = {}

        response = self._kickoff_crew(inputs=kwargs)
        kickoff_id: str | None = response.get("kickoff_id")

        if kickoff_id is None:
            return f"Error: Failed to kickoff crew. Response: {response}"

        # Poll for completion
        for i in range(self.max_polling_time):
            try:
                status_response = self._get_crew_status(crew_id=kickoff_id)
                if status_response.get("state", "").lower() == "success":
                    return str(status_response.get("result", "No result returned"))
                if status_response.get("state", "").lower() == "failed":
                    return f"Error: Crew task failed. Response: {status_response}"
            except Exception as e:
                if i == self.max_polling_time - 1:
                    return f"Error: Failed to get crew status after {self.max_polling_time} attempts. Last error: {e}"

            time.sleep(1)

        return f"Error: Crew did not complete within {self.max_polling_time} seconds"
