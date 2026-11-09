import sys
import types
from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.scrapegraph_scrape_tool import scrapegraph_scrape_tool as module


@pytest.fixture
def sgai_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake ``scrapegraph_py`` package with a spy logger."""
    package = types.ModuleType("scrapegraph_py")
    package.Client = MagicMock()
    logger_module = types.ModuleType("scrapegraph_py.logger")
    logger = MagicMock()
    logger_module.sgai_logger = logger
    package.logger = logger_module
    monkeypatch.setitem(sys.modules, "scrapegraph_py", package)
    monkeypatch.setitem(sys.modules, "scrapegraph_py.logger", logger_module)
    return logger


def test_enable_logging_is_honored(sgai_logger: MagicMock) -> None:
    """The constructor option must turn logging on.

    ``enable_logging`` was accepted as a parameter but never assigned to the
    model field, so the check below it always saw the ``False`` default.
    """
    tool = module.ScrapegraphScrapeTool(api_key="sgai-test", enable_logging=True)

    assert tool.enable_logging is True
    sgai_logger.set_logging.assert_called_once_with(level="INFO")


def test_logging_is_off_by_default(sgai_logger: MagicMock) -> None:
    tool = module.ScrapegraphScrapeTool(api_key="sgai-test")

    assert tool.enable_logging is False
    sgai_logger.set_logging.assert_not_called()
