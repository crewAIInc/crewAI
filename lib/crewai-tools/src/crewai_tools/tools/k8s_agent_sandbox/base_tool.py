from typing import Any, Callable, TYPE_CHECKING
import time
import shlex
import logging
from pydantic import (
    BaseModel,
    Field,
    SkipValidation,
)
from abc import abstractmethod

if TYPE_CHECKING:
    from k8s_agent_sandbox.sandbox import Sandbox  # type: ignore[import-untyped]

from crewai.tools import BaseTool

from .toolset import K8sAgentSandboxToolset


logger = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT_SEC = 60

CLEANUP_TIMEOUT_SEC = 10


class K8sAgentSandboxBaseTool(BaseTool, arbitrary_types_allowed=True):
    name: str
    description: str
    toolset: SkipValidation[K8sAgentSandboxToolset] = Field(
        exclude=True, description="The instance of the ``K8sAgentSandboxToolset``."
    )

    def model_post_init(self, __context: Any) -> None:
        self.toolset.add_tool(self)
        super().model_post_init(__context)

    def _run(self, *args: Any, **kwargs: Any) -> BaseModel:
        try:
            sandbox = self.toolset.lifecycle_manager.acquire_sandbox()
            return self._run_with_sandbox(sandbox, *args, **kwargs)
        finally:
            self.toolset.lifecycle_manager.release_sandbox()

    @abstractmethod
    def _run_with_sandbox(  # type: ignore[no-any-unimported]
        self, sandbox: "Sandbox", *args: Any, **kwargs: Any
    ) -> BaseModel:
        pass


def remove_staged_file(sandbox: "Sandbox", path: str) -> None:  # type: ignore[no-any-unimported]
    """
    Removes a temporary file that a tool staged inside the sandbox.

    The tools delete their staging files as a part of the command that consumes
    them, so this only covers the case where that command never got to run. It
    is best effort: a sandbox that cannot be reached anymore is not worth
    masking the original error with.
    """

    try:
        result = sandbox.commands.run(
            f"rm -f -- {shlex.quote(path)}",
            timeout=CLEANUP_TIMEOUT_SEC,
        )
    except Exception:
        logger.warning(
            "Could not remove the temporary file %s from the sandbox.",
            path,
            exc_info=True,
        )
        return

    if result.exit_code != 0:
        logger.warning(
            "Could not remove the temporary file %s from the sandbox. Error: %s.",
            path,
            result.stderr,
        )


def create_timeout_tracker(timeout: int) -> Callable[[], int]:
    """
    This returns a helper tracker to track one global timeout
    in case where we run multiple sandbox commands in one tool action.
    """

    start_time = int(time.time())

    def get_remaining() -> int:
        remaining_time = timeout - (int(time.time()) - start_time)
        if remaining_time <= 0:
            raise TimeoutError("Timeout. Cannot continue the tool action.")
        return remaining_time

    return get_remaining
