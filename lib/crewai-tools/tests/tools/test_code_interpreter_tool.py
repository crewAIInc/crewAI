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


# --- Security fix tests ---


class TestCommandInjectionFix:
    """Tests that library names cannot be used for command injection."""

    def test_unsafe_mode_uses_subprocess_not_os_system(self, printer_mock):
        """Verify that run_code_unsafe uses subprocess.run with list args,
        not os.system with string interpolation."""
        tool = CodeInterpreterTool(unsafe_mode=True)
        code = "result = 1 + 1"
        malicious_library = "numpy; echo pwned"

        with patch(
            "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run"
        ) as subprocess_mock:
            tool.run_code_unsafe(code, [malicious_library])

            # subprocess.run should be called with a list, not a shell string
            subprocess_mock.assert_called_once_with(
                ["pip", "install", malicious_library],
                check=True,
            )

    def test_unsafe_mode_no_os_system_call(self, printer_mock):
        """Verify os.system is never called during library installation."""
        tool = CodeInterpreterTool(unsafe_mode=True)
        code = "result = 1 + 1"

        with patch(
            "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.os.system"
        ) as os_system_mock, patch(
            "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run"
        ):
            tool.run_code_unsafe(code, ["numpy"])
            os_system_mock.assert_not_called()

    def test_unsafe_mode_installs_multiple_libraries_safely(self, printer_mock):
        """Verify each library is installed as a separate subprocess call."""
        tool = CodeInterpreterTool(unsafe_mode=True)
        code = "result = 1 + 1"
        libraries = ["numpy", "pandas", "requests"]

        with patch(
            "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run"
        ) as subprocess_mock:
            tool.run_code_unsafe(code, libraries)

            assert subprocess_mock.call_count == 3
            for lib in libraries:
                subprocess_mock.assert_any_call(
                    ["pip", "install", lib],
                    check=True,
                )


class TestSandboxEscapeFix:
    """Tests that sandbox escape via __subclasses__ introspection is blocked."""

    def test_subclasses_introspection_blocked(
        self, printer_mock, docker_unavailable_mock
    ):
        """The classic sandbox escape via __subclasses__() must be blocked."""
        tool = CodeInterpreterTool()
        code = """
for c in ().__class__.__bases__[0].__subclasses__():
    if c.__name__ == 'BuiltinImporter':
        result = c.load_module('os').system('id')
        break
"""
        result = tool.run(code=code, libraries_used=[])
        assert "not allowed in the sandbox" in result

    def test_class_attr_blocked(self, printer_mock, docker_unavailable_mock):
        """Direct __class__ access must be blocked."""
        tool = CodeInterpreterTool()
        code = 'result = "".__class__'
        result = tool.run(code=code, libraries_used=[])
        assert "not allowed in the sandbox" in result

    def test_bases_attr_blocked(self, printer_mock, docker_unavailable_mock):
        """Direct __bases__ access must be blocked."""
        tool = CodeInterpreterTool()
        code = "result = str.__bases__"
        result = tool.run(code=code, libraries_used=[])
        assert "not allowed in the sandbox" in result

    def test_mro_attr_blocked(self, printer_mock, docker_unavailable_mock):
        """Direct __mro__ access must be blocked."""
        tool = CodeInterpreterTool()
        code = "result = str.__mro__"
        result = tool.run(code=code, libraries_used=[])
        assert "not allowed in the sandbox" in result

    def test_subclasses_attr_blocked(self, printer_mock, docker_unavailable_mock):
        """Direct __subclasses__ access must be blocked."""
        tool = CodeInterpreterTool()
        code = "result = object.__subclasses__()"
        result = tool.run(code=code, libraries_used=[])
        assert "not allowed in the sandbox" in result

    def test_getattr_blocked_in_sandbox(self, printer_mock, docker_unavailable_mock):
        """getattr() must be blocked to prevent attribute access bypass."""
        tool = CodeInterpreterTool()
        code = """
result = getattr(str, "__bases__")
"""
        result = tool.run(code=code, libraries_used=[])
        # Should be blocked by either the restricted attrs check or the builtins removal
        assert "not allowed" in result or "not defined" in result

    def test_safe_code_still_works(self, printer_mock, docker_unavailable_mock):
        """Normal code without introspection must still execute correctly."""
        tool = CodeInterpreterTool()
        code = """
x = [1, 2, 3]
result = sum(x) * 2
"""
        result = tool.run(code=code, libraries_used=[])
        assert result == 12

    def test_restricted_attrs_check_method(self):
        """Test _check_restricted_attrs directly."""
        # Safe code should pass
        SandboxPython._check_restricted_attrs("result = 2 + 2")

        # Each restricted attr should raise
        for attr in SandboxPython.RESTRICTED_ATTRS:
            with pytest.raises(RuntimeError, match="not allowed in the sandbox"):
                SandboxPython._check_restricted_attrs(f"x = obj.{attr}")
