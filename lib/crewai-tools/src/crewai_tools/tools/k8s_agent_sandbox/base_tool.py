from typing import Any, Callable, Optional
import time
import logging
from pydantic import (
    Field,
    SkipValidation,
)
from abc import abstractmethod

from pydantic import Field
from crewai.tools import BaseTool

from .toolset import K8sAgentSandboxToolset


logger = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT_SEC = 60


class K8sAgentSandboxBaseTool(BaseTool, arbitrary_types_allowed=True):
    name: str
    description: str
    toolset: SkipValidation[K8sAgentSandboxToolset] = Field(
        exclude=True, description="The instance of the ``K8sAgentSandboxToolset``."
    )

    def model_post_init(self, __context: Any) -> None:
        self.toolset.add_tool(self)
        super().model_post_init(__context)

    def _run(self, *args, **kwargs: Any) -> dict[str, Any]:
        try:
            sandbox = self.toolset.lifecycle_manager.acquire_sandbox()
            return self._run_with_sandbox(sandbox, *args, **kwargs)
        finally:
            self.toolset.lifecycle_manager.release_sandbox()

    @abstractmethod
    def _run_with_sandbox(self, sandbox, *args, **kwargs) -> dict[str, Any]:
        pass


def create_timeout_tracker(timeout: int) -> Callable[[], int]:
    """
    This returns a helper tracker to track one global timeout
    in case where we run multple sandbox commands in one tool action.
    """

    start_time = int(time.time())

    def get_remaining():
        remaining_time = timeout - (int(time.time()) - start_time)
        if remaining_time < 0:
            raise TimeoutError("Timeout. Cannot continue the tool action.")
        return remaining_time if remaining_time >= 0 else 0

    return get_remaining
