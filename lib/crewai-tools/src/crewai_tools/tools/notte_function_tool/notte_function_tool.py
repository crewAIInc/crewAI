"""NotteFunctionTool invokes a pre-deployed Notte Function (serverless workflow).

A Notte Function is a script that has been deployed on the Notte platform as an
invokable API endpoint. Functions are parameterised, versioned, and can be
scheduled. This tool lets a CrewAI agent call any Function the user already
owns by passing its `function_id` and a `variables` dictionary at run time.

Where `NotteBrowserTool` drives a live browser session, `NotteFunctionTool`
delegates to a hardened, pre-tested workflow whose code already lives on
Notte's infrastructure. The two are complementary: agents use the browser tool
for ad-hoc exploration and the function tool for production-grade execution
of known tasks.

Authentication: `NOTTE_API_KEY` environment variable, read automatically by
the SDK.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


# Conditional import so the module can be imported even when the optional
# `notte-sdk` dependency is not installed.
_HAS_NOTTE = False

try:
    from notte_sdk import NotteClient  # type: ignore[import-not-found]

    _HAS_NOTTE = True
except ImportError:
    NotteClient = Any  # type: ignore[assignment, misc]


class NotteFunctionResult(BaseModel):
    """Result returned by `NotteFunctionTool`."""

    success: bool = Field(
        ..., description="Whether the function run completed successfully"
    )
    data: str | dict[str, Any] | list[Any] = Field(
        ..., description="The output payload from the function run"
    )
    error: str | None = Field(
        None, description="Optional error message if the run failed"
    )
    function_run_id: str | None = Field(
        None, description="The Notte function-run identifier, useful for retrieval"
    )


class NotteFunctionToolSchema(BaseModel):
    """Input schema for `NotteFunctionTool`."""

    variables: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Dictionary of input variables passed to the deployed Notte Function. "
            "Keys and types must match the variables expected by the function "
            "definition. Pass `{}` or omit the field for parameter-less functions."
        ),
    )


class NotteFunctionTool(BaseTool):
    """Invoke a pre-deployed Notte Function (serverless workflow) from a CrewAI agent.

    Notte Functions (https://docs.notte.cc) are scripts deployed on Notte as
    invokable API endpoints: parameterised, versioned, schedulable, and tested
    once rather than re-derived by an LLM every run. They are designed for
    production-grade execution of known tasks, in contrast with the ad-hoc
    natural-language action loop exposed by `NotteBrowserTool`.

    The tool binds to a single `function_id` at construction time so the agent
    can simply choose when to invoke it and which `variables` to pass. The
    function definition itself stays in source-controlled, reviewable code on
    the Notte platform, not in the agent's prompt.

    Usage:
        from crewai import Agent, Crew, Task
        from crewai_tools import NotteFunctionTool

        scrape_pricing = NotteFunctionTool(
            function_id="fn_abc123",
            description=(
                "Run the deployed `scrape-pricing` Notte function. "
                "Takes `vendor` and `tier` as input variables."
            ),
        )

        researcher = Agent(
            role="Pricing Analyst",
            goal="Compile competitor pricing into a single report",
            backstory="I monitor SaaS pricing for a living.",
            tools=[scrape_pricing],
        )
    """

    name: str = "Notte Function Tool"
    description: str = (
        "Invoke a pre-deployed Notte Function (serverless workflow). The "
        "function is bound at construction via `function_id`. At call time, "
        "pass a `variables` dict whose keys match the function's expected "
        "inputs. Returns the function-run output as a JSON-encoded result."
    )
    args_schema: type[BaseModel] = NotteFunctionToolSchema

    package_dependencies: ClassVar[list[str]] = ["notte-sdk"]
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="NOTTE_API_KEY",
                description="API key for Notte (sign up at https://notte.cc).",
                required=True,
            ),
        ]
    )

    # Configuration
    function_id: str = ""
    api_key: str | None = None
    decryption_key: str | None = None
    timeout: int | None = None
    raise_on_failure: bool = False

    # Runtime
    _client: Any = None
    _testing: bool = False

    def __init__(
        self,
        function_id: str,
        api_key: str | None = None,
        decryption_key: str | None = None,
        timeout: int | None = None,
        raise_on_failure: bool = False,
        _testing: bool = False,
        **kwargs: Any,
    ) -> None:
        self._testing = _testing
        super().__init__(**kwargs)

        if not function_id:
            raise ValueError(
                "function_id is required: pass the id of a deployed Notte Function."
            )
        self.function_id = function_id
        self.api_key = api_key or os.getenv("NOTTE_API_KEY")
        self.decryption_key = decryption_key
        self.timeout = timeout
        self.raise_on_failure = raise_on_failure

        self._check_required_credentials()

    def _check_required_credentials(self) -> None:
        """Validate that the SDK is installed and credentials are present."""
        if not self._testing and not _HAS_NOTTE:
            raise ImportError(
                "`notte-sdk` package not found. "
                "Install with `pip install notte-sdk` or `uv add notte-sdk`."
            )
        if not self.api_key:
            raise ValueError(
                "api_key is required (or set NOTTE_API_KEY in the environment). "
                "Sign up at https://notte.cc to get one."
            )

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = NotteClient(api_key=self.api_key)
        return self._client

    def _run(
        self,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Invoke the bound Notte Function with the supplied variables.

        Returns a JSON-encoded `NotteFunctionResult`.
        """
        try:
            client = self._ensure_client()
        except Exception as exc:  # noqa: BLE001
            return NotteFunctionResult(
                success=False,
                data="",
                error=f"Failed to initialise Notte client: {exc}",
            ).model_dump_json()

        kwargs = variables or {}
        try:
            workflow = client.Workflow(
                self.function_id, decryption_key=self.decryption_key
            )
            response = workflow.run(
                timeout=self.timeout,
                raise_on_failure=self.raise_on_failure,
                **kwargs,
            )

            success = bool(getattr(response, "success", False))
            run_id = (
                getattr(response, "workflow_run_id", None)
                or getattr(response, "run_id", None)
            )

            data_field = getattr(response, "data", None)
            if data_field is None:
                # Fall back to a model dump if the response carries no `data`.
                if hasattr(response, "model_dump"):
                    data_field = response.model_dump()
                else:
                    data_field = str(response)

            return NotteFunctionResult(
                success=success,
                data=data_field,
                error=None if success else "Function run did not complete successfully.",
                function_run_id=run_id,
            ).model_dump_json()

        except Exception as exc:  # noqa: BLE001
            return NotteFunctionResult(
                success=False,
                data="",
                error=f"{type(exc).__name__}: {exc}",
            ).model_dump_json()
