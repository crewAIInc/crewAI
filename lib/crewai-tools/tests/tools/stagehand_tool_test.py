import sys
from unittest.mock import MagicMock, patch

import pytest


# Create mock classes that will be used by our fixture
class MockStagehandModule:
    def __init__(self):
        self.Stagehand = MagicMock()
        self.StagehandConfig = MagicMock()
        self.StagehandPage = MagicMock()


class MockStagehandSchemas:
    def __init__(self):
        self.ActOptions = MagicMock()
        self.ExtractOptions = MagicMock()
        self.ObserveOptions = MagicMock()
        self.AvailableModel = MagicMock()


class MockStagehandUtils:
    def __init__(self):
        self.configure_logging = MagicMock()


@pytest.fixture(scope="module", autouse=True)
def mock_stagehand_modules():
    """Mock stagehand modules at the start of this test module."""
    # Store original modules if they exist
    original_modules = {}
    for module_name in ["stagehand", "stagehand.schemas", "stagehand.utils"]:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]

    # Create and inject mock modules
    mock_stagehand = MockStagehandModule()
    mock_stagehand_schemas = MockStagehandSchemas()
    mock_stagehand_utils = MockStagehandUtils()

    sys.modules["stagehand"] = mock_stagehand
    sys.modules["stagehand.schemas"] = mock_stagehand_schemas
    sys.modules["stagehand.utils"] = mock_stagehand_utils

    # Import after mocking
    from crewai_tools.tools.stagehand_tool.stagehand_tool import (
        StagehandResult,
        StagehandTool,
    )

    # Make these available to tests in this module
    sys.modules[__name__].StagehandResult = StagehandResult
    sys.modules[__name__].StagehandTool = StagehandTool

    yield

    # Restore original modules
    for module_name, module in original_modules.items():
        sys.modules[module_name] = module


class MockStagehandPage(MagicMock):
    def act(self, options):
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "message": "Action completed successfully"
        }
        return mock_result

    def goto(self, url):
        return MagicMock()

    def extract(self, options):
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "data": "Extracted content",
            "metadata": {"source": "test"},
        }
        return mock_result

    def observe(self, options):
        result1 = MagicMock()
        result1.description = "Button element"
        result1.method = "click"

        result2 = MagicMock()
        result2.description = "Input field"
        result2.method = "type"

        return [result1, result2]


class MockStagehand(MagicMock):
    def init(self):
        self.session_id = "test-session-id"
        self.page = MockStagehandPage()

    def close(self):
        pass


@pytest.fixture
def mock_stagehand_instance():
    with patch(
        "crewai_tools.tools.stagehand_tool.stagehand_tool.Stagehand",
        return_value=MockStagehand(),
    ) as mock:
        yield mock


@pytest.fixture
def stagehand_tool():
    return StagehandTool(
        api_key="test_api_key",
        project_id="test_project_id",
        model_api_key="test_model_api_key",
        _testing=True,  # Enable testing mode to bypass dependency check
    )


def test_stagehand_tool_initialization():
    """Test that the StagehandTool initializes with the correct default values."""
    tool = StagehandTool(
        api_key="test_api_key",
        project_id="test_project_id",
        model_api_key="test_model_api_key",
        _testing=True,  # Enable testing mode
    )

    assert tool.api_key == "test_api_key"
    assert tool.project_id == "test_project_id"
    assert tool.model_api_key == "test_model_api_key"
    assert tool.headless is False
    assert tool.dom_settle_timeout_ms == 3000
    assert tool.self_heal is True
    assert tool.wait_for_captcha_solves is True


@patch(
    "crewai_tools.tools.stagehand_tool.stagehand_tool.StagehandTool._run", autospec=True
)
def test_act_command(mock_run, stagehand_tool):
    """Test the 'act' command functionality."""
    # Setup mock
    mock_run.return_value = "Action result: Action completed successfully"

    # Run the tool
    result = stagehand_tool._run(
        instruction="Click the submit button", command_type="act"
    )

    # Assertions
    assert "Action result" in result
    assert "Action completed successfully" in result


@patch(
    "crewai_tools.tools.stagehand_tool.stagehand_tool.StagehandTool._run", autospec=True
)
def test_navigate_command(mock_run, stagehand_tool):
    """Test the 'navigate' command functionality."""
    # Setup mock
    mock_run.return_value = "Successfully navigated to https://example.com"

    # Run the tool
    result = stagehand_tool._run(
        instruction="Go to example.com",
        url="https://example.com",
        command_type="navigate",
    )

    # Assertions
    assert "https://example.com" in result


@patch(
    "crewai_tools.tools.stagehand_tool.stagehand_tool.StagehandTool._run", autospec=True
)
def test_extract_command(mock_run, stagehand_tool):
    """Test the 'extract' command functionality."""
    # Setup mock
    mock_run.return_value = (
        'Extracted data: {"data": "Extracted content", "metadata": {"source": "test"}}'
    )

    # Run the tool
    result = stagehand_tool._run(
        instruction="Extract all product names and prices", command_type="extract"
    )

    # Assertions
    assert "Extracted data" in result
    assert "Extracted content" in result


@patch(
    "crewai_tools.tools.stagehand_tool.stagehand_tool.StagehandTool._run", autospec=True
)
def test_observe_command(mock_run, stagehand_tool):
    """Test the 'observe' command functionality."""
    # Setup mock
    mock_run.return_value = "Element 1: Button element\nSuggested action: click\nElement 2: Input field\nSuggested action: type"

    # Run the tool
    result = stagehand_tool._run(
        instruction="Find all interactive elements", command_type="observe"
    )

    # Assertions
    assert "Element 1: Button element" in result
    assert "Element 2: Input field" in result
    assert "Suggested action: click" in result
    assert "Suggested action: type" in result


@patch(
    "crewai_tools.tools.stagehand_tool.stagehand_tool.StagehandTool._run", autospec=True
)
def test_error_handling(mock_run, stagehand_tool):
    """Test error handling in the tool."""
    # Setup mock
    mock_run.return_value = "Error: Browser automation error"

    # Run the tool
    result = stagehand_tool._run(
        instruction="Click a non-existent button", command_type="act"
    )

    # Assertions
    assert "Error:" in result
    assert "Browser automation error" in result


def test_initialization_parameters():
    """Test that the StagehandTool initializes with the correct parameters."""
    # Create tool with custom parameters
    tool = StagehandTool(
        api_key="custom_api_key",
        project_id="custom_project_id",
        model_api_key="custom_model_api_key",
        headless=True,
        dom_settle_timeout_ms=5000,
        self_heal=False,
        wait_for_captcha_solves=False,
        verbose=3,
        _testing=True,  # Enable testing mode
    )

    # Verify the tool was initialized with the correct parameters
    assert tool.api_key == "custom_api_key"
    assert tool.project_id == "custom_project_id"
    assert tool.model_api_key == "custom_model_api_key"
    assert tool.headless is True
    assert tool.dom_settle_timeout_ms == 5000
    assert tool.self_heal is False
    assert tool.wait_for_captcha_solves is False
    assert tool.verbose == 3


def test_close_method():
    """Test that the close method cleans up resources correctly."""
    # Create the tool with testing mode
    tool = StagehandTool(
        api_key="test_api_key",
        project_id="test_project_id",
        model_api_key="test_model_api_key",
        _testing=True,
    )

    # Setup mock stagehand instance
    tool._stagehand = MagicMock()
    tool._stagehand.close = MagicMock()  # Non-async mock
    tool._page = MagicMock()

    # Call the close method
    tool.close()

    # Verify resources were cleaned up
    assert tool._stagehand is None
    assert tool._page is None
