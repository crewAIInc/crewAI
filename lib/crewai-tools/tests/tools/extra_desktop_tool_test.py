import json
import sys
from unittest.mock import MagicMock, patch
import pytest

from crewai_tools import ExtraDesktopTool


@pytest.fixture
def tool():
    return ExtraDesktopTool()


def test_tool_attributes(tool):
    assert tool.name == "Extra Desktop Automation Tool"
    assert "Sovereign desktop automation" in tool.description
    assert tool.package_dependencies == ["extra-desktop"]


def test_missing_extra_dependency(tool):
    with patch.dict(sys.modules, {"extra": None}):
        result = tool._run(action="launch", app_name="calc")
        assert "Error: 'extra-desktop' is not installed" in result


def test_launch_action(tool):
    mock_extra = MagicMock()
    mock_shell = MagicMock()
    mock_shell.launch_application.return_value = {"pid": 1234, "status": "running"}

    with patch.dict(
        sys.modules,
        {
            "extra": mock_extra,
            "extra.fastpath": MagicMock(),
            "extra.fastpath.shell": mock_shell,
            "extra.core": MagicMock(),
            "extra.core.input_engine": MagicMock(),
            "extra.core.capture": MagicMock(),
            "extra.core.focus": MagicMock(),
        },
    ):
        result = tool._run(action="launch", app_name="calc")
        data = json.loads(result)
        assert data["status"] == "launched"
        assert data["app"] == "calc"
        mock_shell.launch_application.assert_called_once_with("calc", args=[])


def test_click_action(tool):
    mock_extra = MagicMock()
    mock_input = MagicMock()
    mock_input.mouse_click.return_value = {"x": 100, "y": 200, "success": True}

    with patch.dict(
        sys.modules,
        {
            "extra": mock_extra,
            "extra.fastpath": MagicMock(),
            "extra.fastpath.shell": MagicMock(),
            "extra.core": MagicMock(),
            "extra.core.input_engine": mock_input,
            "extra.core.capture": MagicMock(),
            "extra.core.focus": MagicMock(),
        },
    ):
        result = tool._run(action="click", x=100, y=200, button="left")
        data = json.loads(result)
        assert data["status"] == "clicked"
        assert data["coords"] == [100, 200]
        mock_input.mouse_click.assert_called_once_with(x=100, y=200, button="left")


def test_type_action(tool):
    mock_extra = MagicMock()
    mock_input = MagicMock()
    mock_input.instant_type.return_value = True

    with patch.dict(
        sys.modules,
        {
            "extra": mock_extra,
            "extra.fastpath": MagicMock(),
            "extra.fastpath.shell": MagicMock(),
            "extra.core": MagicMock(),
            "extra.core.input_engine": mock_input,
            "extra.core.capture": MagicMock(),
            "extra.core.focus": MagicMock(),
        },
    ):
        result = tool._run(action="type", text="hello world", press_enter=True)
        data = json.loads(result)
        assert data["status"] == "typed"
        assert data["length"] == 11
        assert data["press_enter"] is True
        mock_input.instant_type.assert_called_once_with("hello world", press_enter=True)


def test_hotkey_action(tool):
    mock_extra = MagicMock()
    mock_input = MagicMock()
    mock_input.send_hotkey.return_value = True

    with patch.dict(
        sys.modules,
        {
            "extra": mock_extra,
            "extra.fastpath": MagicMock(),
            "extra.fastpath.shell": MagicMock(),
            "extra.core": MagicMock(),
            "extra.core.input_engine": mock_input,
            "extra.core.capture": MagicMock(),
            "extra.core.focus": MagicMock(),
        },
    ):
        result = tool._run(action="hotkey", keys=["ctrl", "s"])
        data = json.loads(result)
        assert data["status"] == "hotkey_sent"
        assert data["keys"] == ["ctrl", "s"]
        mock_input.send_hotkey.assert_called_once_with(["ctrl", "s"])


def test_screenshot_action(tool):
    mock_extra = MagicMock()
    mock_capture = MagicMock()
    mock_capture.capture_screen.return_value = "/tmp/screenshot.png"

    with patch.dict(
        sys.modules,
        {
            "extra": mock_extra,
            "extra.fastpath": MagicMock(),
            "extra.fastpath.shell": MagicMock(),
            "extra.core": MagicMock(),
            "extra.core.input_engine": MagicMock(),
            "extra.core.capture": mock_capture,
            "extra.core.focus": MagicMock(),
        },
    ):
        result = tool._run(action="screenshot")
        data = json.loads(result)
        assert data["status"] == "captured"
        assert "screenshot.png" in data["screenshot_path"]
        mock_capture.capture_screen.assert_called_once_with(save_to_file=True)


def test_snap_layout_action(tool):
    mock_extra = MagicMock()
    mock_focus = MagicMock()
    mock_focus.snap_layout.return_value = True

    with patch.dict(
        sys.modules,
        {
            "extra": mock_extra,
            "extra.fastpath": MagicMock(),
            "extra.fastpath.shell": MagicMock(),
            "extra.core": MagicMock(),
            "extra.core.input_engine": MagicMock(),
            "extra.core.capture": MagicMock(),
            "extra.core.focus": mock_focus,
        },
    ):
        result = tool._run(
            action="snap_layout",
            layout="side_by_side",
            left_window="Left",
            right_window="Right",
        )
        data = json.loads(result)
        assert data["status"] == "snapped"
        mock_focus.snap_layout.assert_called_once_with(
            layout="side_by_side", left_window="Left", right_window="Right"
        )


def test_unknown_action(tool):
    mock_extra = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "extra": mock_extra,
            "extra.fastpath": MagicMock(),
            "extra.fastpath.shell": MagicMock(),
            "extra.core": MagicMock(),
            "extra.core.input_engine": MagicMock(),
            "extra.core.capture": MagicMock(),
            "extra.core.focus": MagicMock(),
        },
    ):
        result = tool._run(action="invalid_action")
        assert "Error: Unknown action" in result
