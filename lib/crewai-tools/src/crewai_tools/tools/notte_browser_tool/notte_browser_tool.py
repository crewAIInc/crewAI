"""NotteBrowserTool drives a managed remote Notte browser session for CrewAI agents.

Wraps the Notte Python SDK to expose four action types via a single tool with
a `command_type` parameter: navigate, act, extract, observe. Mirrors the
single-tool / command_type shape used by `StagehandTool`.

Authentication: `NOTTE_API_KEY` environment variable, read automatically by
the SDK.
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


# Conditional import so the module can be imported even when the optional
# `notte-sdk` dependency is not installed.
_HAS_NOTTE = False

try:
    from notte_sdk import Goto, NotteClient  # type: ignore[import-not-found]

    _HAS_NOTTE = True
except ImportError:
    Goto = Any  # type: ignore[assignment, misc]
    NotteClient = Any  # type: ignore[assignment, misc]


class NotteBrowserResult(BaseModel):
    """Result returned by `NotteBrowserTool`.

    Attributes:
        success: Whether the operation completed successfully.
        data: The result payload (string for navigate / act, structured for
            extract / observe).
        error: Optional error message if the operation failed.
    """

    success: bool = Field(
        ..., description="Whether the operation completed successfully"
    )
    data: str | dict[str, Any] | list[Any] = Field(
        ..., description="The result payload from the operation"
    )
    error: str | None = Field(
        None, description="Optional error message if the operation failed"
    )


class NotteBrowserToolSchema(BaseModel):
    """Input schema for `NotteBrowserTool`."""

    instruction: str | None = Field(
        None,
        description=(
            "Natural-language instruction for the browser. "
            "For 'act', describe a single action (e.g. 'click the sign-up button'). "
            "For 'extract', describe what to pull out of the page. "
            "For 'observe', leave empty to list all interactive elements. "
            "For 'navigate', omit when a URL is provided."
        ),
    )
    url: str | None = Field(
        None,
        description="URL to navigate to. Used with command_type='navigate'.",
    )
    command_type: str | None = Field(
        "act",
        description=(
            "Type of command to run (choose one):\n"
            "- 'navigate': go to a URL\n"
            "- 'act': perform a natural-language action on the current page "
            "(default)\n"
            "- 'extract': extract structured data from the current page\n"
            "- 'observe': list interactive elements on the current page"
        ),
    )


class NotteBrowserTool(BaseTool):
    """A tool that drives a managed remote Notte browser session.

    Notte (https://notte.cc) provides hosted browser infrastructure plus a
    perception layer that lets agents interact with web pages using natural
    language. Sessions are remote-hosted, so no local browser is required.

    Supports four command types via a single tool interface:

    1. `navigate`: open a URL in the session
    2. `act`: perform a natural-language action on the current page
    3. `extract`: extract structured data from the current page
    4. `observe`: list the interactive elements on the current page

    Implementation note: `act` is executed via Notte's autonomous agent with
    a tight step budget (default 3), because Notte's session-level act surface
    is typed (action-ID based) rather than free-text. The tool boundary stays
    consistent with `StagehandTool`; the engine underneath differs.

    Usage:
        from crewai import Agent, Crew, Task
        from crewai_tools import NotteBrowserTool

        with NotteBrowserTool() as notte_tool:
            agent = Agent(
                role="Web Researcher",
                goal="Find and summarise information from websites",
                backstory="I am an expert at finding things online.",
                tools=[notte_tool],
            )
            task = Task(
                description=(
                    "Navigate to https://news.ycombinator.com and report the "
                    "titles of the top three posts."
                ),
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task])
            print(crew.kickoff())
    """

    name: str = "Notte Browser Tool"
    description: str = (
        "Drives a managed remote Notte browser session. Supports four "
        "command_types: 'navigate' to a URL, 'act' to perform a natural-"
        "language action, 'extract' structured data from the current page, "
        "and 'observe' interactive elements on the current page."
    )
    args_schema: type[BaseModel] = NotteBrowserToolSchema

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
    api_key: str | None = None
    headless: bool = True
    max_act_steps: int = 3

    # Instance state
    _client: Any = None
    _session: Any = None
    _testing: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        headless: bool = True,
        max_act_steps: int = 3,
        _testing: bool = False,
        **kwargs: Any,
    ) -> None:
        # Set testing flag early so init logic can rely on it.
        self._testing = _testing
        super().__init__(**kwargs)

        self.api_key = api_key or os.getenv("NOTTE_API_KEY")
        self.headless = headless
        self.max_act_steps = max_act_steps

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

    # ----- lifecycle -----

    def _ensure_session(self) -> Any:
        """Lazily create a `NotteClient` and an active `Session`."""
        if self._session is not None:
            return self._session
        if self._client is None:
            self._client = NotteClient(api_key=self.api_key)
        session = self._client.Session(headless=self.headless)
        session.start()
        self._session = session
        return session

    def close(self) -> None:
        """Stop the active session if one is open."""
        try:
            if self._session is not None:
                self._session.stop()
        finally:
            self._session = None

    def __enter__(self) -> NotteBrowserTool:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ----- core runner -----

    def _run(
        self,
        instruction: str | None = None,
        url: str | None = None,
        command_type: str = "act",
    ) -> str:
        """Execute one of: navigate, act, extract, observe.

        Returns a JSON-encoded `NotteBrowserResult`.
        """
        command_type = (command_type or "act").lower()

        try:
            session = self._ensure_session()
        except Exception as exc:  # noqa: BLE001
            return NotteBrowserResult(
                success=False,
                data="",
                error=f"Failed to start Notte session: {exc}",
            ).model_dump_json()

        try:
            if command_type == "navigate":
                target = url or instruction
                if not target:
                    return NotteBrowserResult(
                        success=False,
                        data="",
                        error=(
                            "`url` is required for command_type='navigate'."
                        ),
                    ).model_dump_json()
                result = session.execute(Goto(url=target))
                return NotteBrowserResult(
                    success=bool(getattr(result, "success", True)),
                    data=f"Navigated to {target}.",
                ).model_dump_json()

            if command_type == "act":
                task = instruction or "Perform the next sensible action."
                agent = self._client.Agent(
                    session=session, max_steps=self.max_act_steps
                )
                response = agent.run(task=task)
                success = bool(getattr(response, "success", False))
                answer = getattr(response, "answer", "") or ""
                return NotteBrowserResult(
                    success=success,
                    data=answer,
                    error=None if success else "Action did not complete successfully.",
                ).model_dump_json()

            if command_type == "extract":
                task = instruction or "Extract the main content from the page."
                result = session.scrape(instructions=task)
                payload: str | dict[str, Any] | list[Any]
                if isinstance(result, (dict, list)):
                    payload = result
                else:
                    payload = str(result)
                return NotteBrowserResult(success=True, data=payload).model_dump_json()

            if command_type == "observe":
                observation = session.observe()
                if hasattr(observation, "model_dump"):
                    payload = observation.model_dump()
                elif hasattr(observation, "model_dump_json"):
                    payload = json.loads(observation.model_dump_json())
                else:
                    payload = str(observation)
                return NotteBrowserResult(
                    success=True, data=payload
                ).model_dump_json()

            return NotteBrowserResult(
                success=False,
                data="",
                error=(
                    f"Unknown command_type: {command_type!r}. "
                    "Expected one of: navigate, act, extract, observe."
                ),
            ).model_dump_json()

        except Exception as exc:  # noqa: BLE001
            return NotteBrowserResult(
                success=False,
                data="",
                error=f"{type(exc).__name__}: {exc}",
            ).model_dump_json()
