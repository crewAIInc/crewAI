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


def test_docker_unavailable_raises_error(printer_mock, docker_unavailable_mock):
    """Test that execution fails when Docker is unavailable in safe mode."""
    tool = CodeInterpreterTool()
    code = """
result = 2 + 2
print(result)
"""
    with pytest.raises(RuntimeError) as exc_info:
        tool.run(code=code, libraries_used=[])
    
    assert "Docker is required for safe code execution" in str(exc_info.value)
    assert "sandbox escape" in str(exc_info.value)


def test_restricted_sandbox_running_with_blocked_modules(
    printer_mock, docker_unavailable_mock
):
    """Test that restricted modules cannot be imported when using the deprecated sandbox directly."""
    tool = CodeInterpreterTool()
    restricted_modules = SandboxPython.BLOCKED_MODULES

    for module in restricted_modules:
        code = f"""
import {module}
result = "Import succeeded"
"""
        # Note: run_code_in_restricted_sandbox is deprecated and insecure
        # This test verifies the old behavior but should not be used in production
        result = tool.run_code_in_restricted_sandbox(code)
        
        assert f"An error occurred: Importing '{module}' is not allowed" in result


def test_restricted_sandbox_running_with_blocked_builtins(
    printer_mock, docker_unavailable_mock
):
    """Test that restricted builtins are not available when using the deprecated sandbox directly."""
    tool = CodeInterpreterTool()
    restricted_builtins = SandboxPython.UNSAFE_BUILTINS

    for builtin in restricted_builtins:
        code = f"""
{builtin}("test")
result = "Builtin available"
"""
        # Note: run_code_in_restricted_sandbox is deprecated and insecure
        # This test verifies the old behavior but should not be used in production
        result = tool.run_code_in_restricted_sandbox(code)
        assert f"An error occurred: name '{builtin}' is not defined" in result


def test_restricted_sandbox_running_with_no_result_variable(
    printer_mock, docker_unavailable_mock
):
    """Test behavior when no result variable is set in deprecated sandbox."""
    tool = CodeInterpreterTool()
    code = """
x = 10
"""
    # Note: run_code_in_restricted_sandbox is deprecated and insecure
    # This test verifies the old behavior but should not be used in production
    result = tool.run_code_in_restricted_sandbox(code)
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


@pytest.mark.xfail(
    reason=(
        "run_code_in_restricted_sandbox is known to be vulnerable to sandbox "
        "escape via object introspection. This test encodes the desired secure "
        "behavior (no escape possible) and will start passing once the "
        "vulnerability is fixed or the function is removed."
    )
)
def test_sandbox_escape_vulnerability_demonstration(printer_mock):
    """Demonstrate that the restricted sandbox is vulnerable to escape attacks.
    
    This test shows that an attacker can use Python object introspection to bypass
    the restricted sandbox and access blocked modules like 'os'. This is why the
    sandbox should never be used for untrusted code execution.
    
    NOTE: This test uses the deprecated run_code_in_restricted_sandbox directly
    to demonstrate the vulnerability. In production, Docker is now required.
    """
    tool = CodeInterpreterTool()
    
    # Classic Python sandbox escape via object introspection
    escape_code = """
# Recover the real __import__ function via object introspection
for cls in ().__class__.__bases__[0].__subclasses__():
    if cls.__name__ == 'catch_warnings':
        # Get the real builtins module
        real_builtins = cls()._module.__builtins__
        real_import = real_builtins['__import__']
        # Now we can import os and execute commands
        os = real_import('os')
        # Demonstrate we have escaped the sandbox
        result = "SANDBOX_ESCAPED" if hasattr(os, 'system') else "FAILED"
        break
"""
    
    # The deprecated sandbox is vulnerable to this attack
    result = tool.run_code_in_restricted_sandbox(escape_code)
    
    # Desired behavior: the restricted sandbox should prevent this escape.
    # If this assertion fails, run_code_in_restricted_sandbox remains vulnerable.
    assert result != "SANDBOX_ESCAPED", (
        "The restricted sandbox was bypassed via object introspection. "
        "This indicates run_code_in_restricted_sandbox is still vulnerable and "
        "is why Docker is now required for safe code execution."
    )
