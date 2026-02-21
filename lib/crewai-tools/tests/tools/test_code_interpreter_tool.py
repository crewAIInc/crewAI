from unittest.mock import patch, MagicMock

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


class TestCommandInjectionPrevention:
    """Tests for command injection prevention in unsafe mode (CVE: CWE-78)."""

    @patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run"
    )
    def test_unsafe_mode_uses_subprocess_run_with_list_args(
        self, subprocess_mock, printer_mock
    ):
        """Verify libraries are installed via subprocess.run with list args, not os.system."""
        tool = CodeInterpreterTool(unsafe_mode=True)
        code = "result = 'done'"
        tool.run_code_unsafe(code, ["numpy"])
        subprocess_mock.assert_called_once_with(
            ["pip", "install", "numpy"],
            check=True,
            capture_output=True,
        )

    @patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run"
    )
    def test_unsafe_mode_library_with_shell_metacharacters(
        self, subprocess_mock, printer_mock
    ):
        """Ensure shell metacharacters in library names are not interpreted."""
        tool = CodeInterpreterTool(unsafe_mode=True)
        malicious_lib = "numpy; id #"
        code = "result = 'done'"
        tool.run_code_unsafe(code, [malicious_lib])
        subprocess_mock.assert_called_once_with(
            ["pip", "install", "numpy; id #"],
            check=True,
            capture_output=True,
        )

    @patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run"
    )
    def test_unsafe_mode_library_with_command_substitution(
        self, subprocess_mock, printer_mock
    ):
        """Ensure command substitution in library names is not executed."""
        tool = CodeInterpreterTool(unsafe_mode=True)
        malicious_lib = "numpy && rm -rf /"
        code = "result = 'done'"
        tool.run_code_unsafe(code, [malicious_lib])
        subprocess_mock.assert_called_once_with(
            ["pip", "install", "numpy && rm -rf /"],
            check=True,
            capture_output=True,
        )

    @patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run"
    )
    def test_unsafe_mode_library_with_backtick_injection(
        self, subprocess_mock, printer_mock
    ):
        """Ensure backtick command injection is not executed."""
        tool = CodeInterpreterTool(unsafe_mode=True)
        malicious_lib = "numpy`whoami`"
        code = "result = 'done'"
        tool.run_code_unsafe(code, [malicious_lib])
        subprocess_mock.assert_called_once_with(
            ["pip", "install", "numpy`whoami`"],
            check=True,
            capture_output=True,
        )

    @patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run"
    )
    def test_unsafe_mode_multiple_libraries_installed_separately(
        self, subprocess_mock, printer_mock
    ):
        """Each library is installed in a separate subprocess call."""
        tool = CodeInterpreterTool(unsafe_mode=True)
        code = "result = 'done'"
        tool.run_code_unsafe(code, ["numpy", "pandas"])
        assert subprocess_mock.call_count == 2


class TestSandboxEscapePrevention:
    """Tests for sandbox escape prevention via object introspection (CVE: CWE-94)."""

    def test_sandbox_blocks_class_attribute(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via __class__ introspection."""
        tool = CodeInterpreterTool()
        code = """
result = ().__class__
"""
        result = tool.run(code=code, libraries_used=[])
        assert "Access to '__class__' is not allowed" in result

    def test_sandbox_blocks_bases_attribute(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via __bases__ introspection."""
        tool = CodeInterpreterTool()
        code = """
result = object.__bases__
"""
        result = tool.run(code=code, libraries_used=[])
        assert "Access to '__bases__' is not allowed" in result

    def test_sandbox_blocks_subclasses_method(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via __subclasses__ introspection."""
        tool = CodeInterpreterTool()
        code = """
result = object.__subclasses__()
"""
        result = tool.run(code=code, libraries_used=[])
        assert "Access to '__subclasses__' is not allowed" in result

    def test_sandbox_blocks_mro_attribute(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via __mro__ introspection."""
        tool = CodeInterpreterTool()
        code = """
result = int.__mro__
"""
        result = tool.run(code=code, libraries_used=[])
        assert "Access to '__mro__' is not allowed" in result

    def test_sandbox_blocks_globals_attribute(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via __globals__ introspection."""
        tool = CodeInterpreterTool()
        code = """
def f(): pass
result = f.__globals__
"""
        result = tool.run(code=code, libraries_used=[])
        assert "Access to '__globals__' is not allowed" in result

    def test_sandbox_blocks_builtins_attribute(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via __builtins__ access."""
        tool = CodeInterpreterTool()
        code = """
result = __builtins__
"""
        result = tool.run(code=code, libraries_used=[])
        assert "Access to '__builtins__' is not allowed" in result

    def test_sandbox_blocks_full_introspection_chain(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent the full sandbox escape PoC from the issue."""
        tool = CodeInterpreterTool()
        code = """
for c in ().__class__.__bases__[0].__subclasses__():
    if c.__name__ == 'BuiltinImporter':
        result = c.load_module('os').system('id')
        break
"""
        result = tool.run(code=code, libraries_used=[])
        assert "is not allowed" in result

    def test_sandbox_blocks_getattr_builtin(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via getattr builtin."""
        tool = CodeInterpreterTool()
        code = """
result = getattr(object, '__subclasses__')()
"""
        result = tool.run(code=code, libraries_used=[])
        assert "An error occurred" in result

    def test_sandbox_blocks_type_builtin(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via type builtin."""
        tool = CodeInterpreterTool()
        code = """
result = type('X', (object,), {})
"""
        result = tool.run(code=code, libraries_used=[])
        assert "An error occurred" in result

    def test_sandbox_blocks_code_attribute(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via __code__ attribute."""
        tool = CodeInterpreterTool()
        code = """
def f(): pass
result = f.__code__
"""
        result = tool.run(code=code, libraries_used=[])
        assert "Access to '__code__' is not allowed" in result

    def test_sandbox_allows_normal_code(
        self, printer_mock, docker_unavailable_mock
    ):
        """Ensure normal code still works in sandbox."""
        tool = CodeInterpreterTool()
        code = """
data = [1, 2, 3, 4, 5]
result = sum(data) * 2
"""
        result = tool.run(code=code, libraries_used=[])
        assert result == 30

    def test_sandbox_blocks_reduce_attribute(
        self, printer_mock, docker_unavailable_mock
    ):
        """Prevent sandbox escape via __reduce__ for pickle attacks."""
        tool = CodeInterpreterTool()
        code = """
result = [].__reduce__()
"""
        result = tool.run(code=code, libraries_used=[])
        assert "Access to '__reduce__' is not allowed" in result
