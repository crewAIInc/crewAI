import subprocess
from unittest.mock import patch

import pytest

from crewai_tools.tools.code_interpreter_tool.code_interpreter_tool import (
    CodeInterpreterTool,
)


@pytest.fixture
def printer_mock():
    with patch("crewai_tools.printer.Printer.print") as mock:
        yield mock


@pytest.fixture
def docker_unavailable_mock():
    with patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.CodeInterpreterTool._check_docker_available",
        return_value=False,
    ) as mock:
        yield mock


@patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.docker_from_env")
def test_run_code_in_docker(docker_mock, printer_mock):
    tool = CodeInterpreterTool()
    code = "print('Hello, World!')"
    libraries_used = ["numpy", "pandas"]
    expected_output = "Hello, World!\n"

    docker_mock().containers.run().exec_run().exit_code = 0
    docker_mock().containers.run().exec_run().output = expected_output.encode()

    result = tool.run_code_in_docker(code, libraries_used)
    assert result == expected_output
    printer_mock.assert_called_with(
        "Running code in Docker environment", color="bold_blue"
    )


@patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.docker_from_env")
def test_run_code_in_docker_with_error(docker_mock, printer_mock):
    tool = CodeInterpreterTool()
    code = "print(1/0)"
    libraries_used = ["numpy", "pandas"]
    expected_output = "Something went wrong while running the code: \nZeroDivisionError: division by zero\n"

    docker_mock().containers.run().exec_run().exit_code = 1
    docker_mock().containers.run().exec_run().output = (
        b"ZeroDivisionError: division by zero\n"
    )

    result = tool.run_code_in_docker(code, libraries_used)
    assert result == expected_output
    printer_mock.assert_called_with(
        "Running code in Docker environment", color="bold_blue"
    )


@patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.docker_from_env")
def test_run_code_in_docker_with_script(docker_mock, printer_mock):
    tool = CodeInterpreterTool()
    code = """print("This is line 1")
print("This is line 2")"""
    libraries_used = []
    expected_output = "This is line 1\nThis is line 2\n"

    docker_mock().containers.run().exec_run().exit_code = 0
    docker_mock().containers.run().exec_run().output = expected_output.encode()

    result = tool.run_code_in_docker(code, libraries_used)
    assert result == expected_output
    printer_mock.assert_called_with(
        "Running code in Docker environment", color="bold_blue"
    )


def test_docker_unavailable_fails_safely(printer_mock, docker_unavailable_mock):
    """Test that code execution fails when Docker is unavailable."""
    tool = CodeInterpreterTool()
    code = """
result = 2 + 2
print(result)
"""
    with pytest.raises(RuntimeError) as exc_info:
        tool.run(code=code, libraries_used=[])

    assert "Docker is required for safe code execution" in str(exc_info.value)
    assert printer_mock.called
    call_args = printer_mock.call_args
    assert "SECURITY ERROR" in call_args[0][0]
    assert call_args[1]["color"] == "bold_red"


def test_docker_unavailable_suggests_unsafe_mode(printer_mock, docker_unavailable_mock):
    """Test that error message suggests unsafe_mode as alternative."""
    tool = CodeInterpreterTool()
    code = "result = 1 + 1"

    with pytest.raises(RuntimeError) as exc_info:
        tool.run(code=code, libraries_used=[])

    error_output = printer_mock.call_args[0][0]
    assert "unsafe_mode=True" in error_output
    assert "NOT recommended" in error_output
    assert "docs.crewai.com" in error_output


def test_unsafe_mode_running_with_no_result_variable(
    printer_mock, docker_unavailable_mock
):
    """Test behavior when no result variable is set in unsafe mode."""
    tool = CodeInterpreterTool(unsafe_mode=True)
    code = """
x = 10
"""
    result = tool.run(code=code, libraries_used=[])
    printer_mock.assert_called_with(
        "⚠️  WARNING: Running code in UNSAFE mode - no security controls active!",
        color="bold_red",
    )
    assert result == "No result variable found."


def test_unsafe_mode_running_unsafe_code(printer_mock, docker_unavailable_mock):
    """Test that unsafe mode allows unrestricted code execution."""
    tool = CodeInterpreterTool(unsafe_mode=True)
    code = """
import os
result = eval("5/1")
"""
    result = tool.run(code=code, libraries_used=[])
    printer_mock.assert_called_with(
        "⚠️  WARNING: Running code in UNSAFE mode - no security controls active!",
        color="bold_red",
    )
    assert 5.0 == result


@patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run")
def test_unsafe_mode_library_installation(subprocess_mock, printer_mock, docker_unavailable_mock):
    """Test that unsafe mode properly installs libraries using subprocess."""
    tool = CodeInterpreterTool(unsafe_mode=True)
    code = "result = 42"
    libraries = ["numpy", "pandas"]

    subprocess_mock.return_value = None

    tool.run(code=code, libraries_used=libraries)

    assert subprocess_mock.call_count == 2
    subprocess_mock.assert_any_call(
        ["pip", "install", "numpy"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    subprocess_mock.assert_any_call(
        ["pip", "install", "pandas"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
