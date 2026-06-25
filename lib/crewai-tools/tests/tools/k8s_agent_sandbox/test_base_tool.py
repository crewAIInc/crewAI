from unittest.mock import MagicMock, patch
from typing import Any

from pydantic import BaseModel, Field
import pytest

from crewai_tools.tools.k8s_agent_sandbox.base_tool import K8sAgentSandboxBaseTool


class DummyToolInputSchema(BaseModel):
    test_arg: str = Field(description="some positional argument")
    test_kwarg: str | None = Field(default=None, description="some keyword argument")


class DummyK8sTool(K8sAgentSandboxBaseTool):
    name: str = "dummy_tool"
    description: str = "Dummy testing tool."
    args_schema: type[BaseModel] = DummyToolInputSchema

    def _run_with_sandbox(self, sandbox, *args, **kwargs) -> dict[str, Any]:
        return self._dummy_work(sandbox, *args, **kwargs)

    def _dummy_work(
        self, sandbox, test_arg: str, test_kwarg: str | None = None
    ) -> dict[str, Any]:
        return {
            "sandbox-claim": sandbox.claim_name,
            "arg": test_arg,
            "kwarg": test_kwarg,
        }


def test_tool_added_to_toolset(sample_toolset):
    assert len(sample_toolset.tools) == 0

    _ = DummyK8sTool(
        toolset=sample_toolset,
    )

    assert len(sample_toolset.tools) == 1


def test_run_with_sandbox(
    sample_toolset,
    mock_sandbox,
    lifecycle_mode_sandbox_termination_expected,
):
    claim_name = "some-claim"
    mock_sandbox.claim_name = claim_name

    tool = DummyK8sTool(
        toolset=sample_toolset,
    )

    assert not mock_sandbox.terminate.called

    result = tool.run(
        "some-arg",
        test_kwarg="some-kwarg",
    )

    assert result == {
        "sandbox-claim": claim_name,
        "arg": "some-arg",
        "kwarg": "some-kwarg",
    }

    if lifecycle_mode_sandbox_termination_expected:
        assert mock_sandbox.terminate.called
    else:
        assert not mock_sandbox.terminate.called


def test_sandbox_released_after_error(
    sample_toolset,
    mock_sandbox,
    lifecycle_mode_sandbox_termination_expected,
):
    class FailingTool(DummyK8sTool):
        def _run_with_sandbox(self, sandbox, *args, **kwargs) -> dict[str, Any]:
            raise Exception("some error")

    tool = FailingTool(toolset=sample_toolset)

    assert not mock_sandbox.terminate.called

    with patch.object(
        sample_toolset,
        "lifecycle_manager",
        MagicMock(wraps=sample_toolset.lifecycle_manager),
    ) as m:
        with pytest.raises(Exception, match="some error"):
            tool.run(
                "some-arg",
                test_kwarg="some-kwarg",
            )

    if lifecycle_mode_sandbox_termination_expected:
        assert mock_sandbox.terminate.called
    else:
        assert not mock_sandbox.terminate.called
