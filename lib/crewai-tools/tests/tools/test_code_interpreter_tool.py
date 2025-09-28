from unittest.mock import patch

from crewai_tools.tools.code_interpreter_tool.code_interpreter_tool import (
    CodeInterpreterTool,
    SandboxPython,
)
import pytest


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


def test_restricted_sandbox_basic_code_execution(printer_mock, docker_unavailable_mock):
    """Test basic code execution."""
    tool = CodeInterpreterTool()
    code = """
result = 2 + 2
print(result)
"""
    result = tool.run(code=code, libraries_used=[])
    printer_mock.assert_called_with(
        "Running code in restricted sandbox", color="yellow"
    )
    assert result == 4


def test_restricted_sandbox_running_with_blocked_modules(
    printer_mock, docker_unavailable_mock
):
    """Test that restricted modules cannot be imported."""
    tool = CodeInterpreterTool()
    restricted_modules = SandboxPython.BLOCKED_MODULES

    for module in restricted_modules:
        code = f"""
import {module}
result = "Import succeeded"
"""
        result = tool.run(code=code, libraries_used=[])
        printer_mock.assert_called_with(
            "Running code in restricted sandbox", color="yellow"
        )

        assert f"An error occurred: Importing '{module}' is not allowed" in result


def test_restricted_sandbox_running_with_blocked_builtins(
    printer_mock, docker_unavailable_mock
):
    """Test that restricted builtins are not available."""
    tool = CodeInterpreterTool()
    restricted_builtins = SandboxPython.UNSAFE_BUILTINS

    for builtin in restricted_builtins:
        code = f"""
{builtin}("test")
result = "Builtin available"
"""
        result = tool.run(code=code, libraries_used=[])
        printer_mock.assert_called_with(
            "Running code in restricted sandbox", color="yellow"
        )
        assert f"An error occurred: name '{builtin}' is not defined" in result


def test_restricted_sandbox_running_with_no_result_variable(
    printer_mock, docker_unavailable_mock
):
    """Test behavior when no result variable is set."""
    tool = CodeInterpreterTool()
    code = """
x = 10
"""
    result = tool.run(code=code, libraries_used=[])
    printer_mock.assert_called_with(
        "Running code in restricted sandbox", color="yellow"
    )
    assert result == "No result variable found."


def test_unsafe_mode_running_with_no_result_variable(
    printer_mock, docker_unavailable_mock
):
    """Test behavior when no result variable is set."""
    tool = CodeInterpreterTool(unsafe_mode=True)
    code = """
x = 10
"""
    result = tool.run(code=code, libraries_used=[])
    printer_mock.assert_called_with(
        "WARNING: Running code in unsafe mode", color="bold_magenta"
    )
    assert result == "No result variable found."


def test_unsafe_mode_running_unsafe_code(printer_mock, docker_unavailable_mock):
    """Test behavior when no result variable is set."""
    tool = CodeInterpreterTool(unsafe_mode=True)
    code = """
import os
os.system("ls -la")
result = eval("5/1")
"""
    result = tool.run(code=code, libraries_used=[])
    printer_mock.assert_called_with(
        "WARNING: Running code in unsafe mode", color="bold_magenta"
    )
    assert 5.0 == result
