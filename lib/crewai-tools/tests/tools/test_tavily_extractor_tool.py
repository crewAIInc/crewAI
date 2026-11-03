from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.tavily_extractor_tool import tavily_extractor_tool as module


def test_dependency_install_invokes_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The auto-install path must run ``uv`` with ``pip`` as a subcommand.

    ``["uv pip", "install", ...]`` passed the two words as a single executable
    name, so ``subprocess.run`` raised ``FileNotFoundError`` (which the
    ``CalledProcessError`` handler does not catch) when the user accepted the
    installation prompt.
    """
    monkeypatch.setattr(module, "TAVILY_AVAILABLE", False)
    monkeypatch.setattr("click.confirm", lambda *args, **kwargs: True)
    run = MagicMock()
    monkeypatch.setattr("subprocess.run", run)

    with pytest.raises(ImportError):
        module.TavilyExtractorTool()

    assert run.call_args.args[0] == ["uv", "pip", "install", "tavily-python"]
