import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def sandlock_unavailable_mock():
    with patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.CodeInterpreterTool._check_sandlock_available",
        return_value=False,
    ) as mock:
        yield mock


@pytest.fixture
def sandlock_available_mock():
    with patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.CodeInterpreterTool._check_sandlock_available",
        return_value=True,
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


def test_docker_and_sandlock_unavailable_raises_error(
    printer_mock, docker_unavailable_mock, sandlock_unavailable_mock
):
    """Test that execution fails when both Docker and sandlock are unavailable."""
    tool = CodeInterpreterTool()
    code = """
result = 2 + 2
print(result)
"""
    with pytest.raises(RuntimeError) as exc_info:
        tool.run(code=code, libraries_used=[])

    assert "No secure execution backend is available" in str(exc_info.value)
    assert "sandlock" in str(exc_info.value)


def test_restricted_sandbox_running_with_blocked_modules():
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


def test_restricted_sandbox_running_with_blocked_builtins():
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


@patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run")
def test_unsafe_mode_installs_libraries_without_shell(
    subprocess_run_mock, printer_mock, docker_unavailable_mock
):
    """Test that library installation uses subprocess.run with shell=False, not os.system."""
    tool = CodeInterpreterTool(unsafe_mode=True)
    code = "result = 1"
    libraries_used = ["numpy", "pandas"]

    tool.run(code=code, libraries_used=libraries_used)

    assert subprocess_run_mock.call_count == 2
    for call, library in zip(subprocess_run_mock.call_args_list, libraries_used):
        args, kwargs = call
        # Must be list form (no shell expansion possible)
        assert args[0] == [sys.executable, "-m", "pip", "install", library]
        # shell= must not be True (defaults to False)
        assert kwargs.get("shell", False) is False


@patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run")
def test_unsafe_mode_library_name_with_shell_metacharacters_does_not_invoke_shell(
    subprocess_run_mock, printer_mock, docker_unavailable_mock
):
    """Test that a malicious library name cannot inject shell commands."""
    tool = CodeInterpreterTool(unsafe_mode=True)
    code = "result = 1"
    malicious_library = "numpy; rm -rf /"

    tool.run(code=code, libraries_used=[malicious_library])

    subprocess_run_mock.assert_called_once()
    args, kwargs = subprocess_run_mock.call_args
    # The entire malicious string is passed as a single argument — no shell parsing
    assert args[0] == [sys.executable, "-m", "pip", "install", malicious_library]
    assert kwargs.get("shell", False) is False


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


# --- Sandlock backend tests ---


def test_sandlock_fallback_when_docker_unavailable(
    printer_mock, docker_unavailable_mock, sandlock_available_mock
):
    """Test that sandlock is used as fallback when Docker is unavailable."""
    tool = CodeInterpreterTool()
    code = "print('hello')"

    with patch.object(
        CodeInterpreterTool,
        "run_code_in_sandlock",
        return_value="hello\n",
    ) as sandlock_run_mock:
        result = tool.run(code=code, libraries_used=[])

    assert result == "hello\n"
    sandlock_run_mock.assert_called_once_with(code, [])


def test_execution_backend_sandlock_calls_sandlock(
    printer_mock, sandlock_available_mock
):
    """Test that execution_backend='sandlock' routes to sandlock."""
    tool = CodeInterpreterTool(execution_backend="sandlock")
    code = "print('test')"

    with patch.object(
        CodeInterpreterTool,
        "run_code_in_sandlock",
        return_value="test\n",
    ) as mock_sandlock:
        result = tool.run(code=code, libraries_used=[])

    assert result == "test\n"
    mock_sandlock.assert_called_once_with(code, [])


def test_execution_backend_docker_calls_docker(printer_mock):
    """Test that execution_backend='docker' routes directly to Docker."""
    tool = CodeInterpreterTool(execution_backend="docker")
    code = "print('test')"

    with patch.object(
        CodeInterpreterTool,
        "run_code_in_docker",
        return_value="test\n",
    ) as mock_docker:
        result = tool.run(code=code, libraries_used=[])

    assert result == "test\n"
    mock_docker.assert_called_once_with(code, [])


def test_execution_backend_unsafe_calls_unsafe(printer_mock):
    """Test that execution_backend='unsafe' routes to unsafe mode."""
    tool = CodeInterpreterTool(execution_backend="unsafe")
    code = "result = 42"

    result = tool.run(code=code, libraries_used=[])
    assert result == 42


def test_sandlock_check_not_linux(printer_mock):
    """Test that sandlock is unavailable on non-Linux systems."""
    with patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.platform.system",
        return_value="Darwin",
    ):
        assert CodeInterpreterTool._check_sandlock_available() is False


def test_sandlock_check_not_installed(printer_mock):
    """Test that sandlock is unavailable when not installed."""
    with patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.platform.system",
        return_value="Linux",
    ):
        with patch(
            "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.importlib.util.find_spec",
            return_value=None,
        ):
            assert CodeInterpreterTool._check_sandlock_available() is False


def test_sandlock_check_available_on_linux(printer_mock):
    """Test that sandlock is available on Linux when installed."""
    with patch(
        "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.platform.system",
        return_value="Linux",
    ):
        with patch(
            "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.importlib.util.find_spec",
            return_value=MagicMock(),  # non-None means installed
        ):
            assert CodeInterpreterTool._check_sandlock_available() is True


def test_sandlock_run_raises_when_unavailable(printer_mock):
    """Test that run_code_in_sandlock raises RuntimeError when sandlock is unavailable."""
    tool = CodeInterpreterTool()
    with patch.object(
        CodeInterpreterTool, "_check_sandlock_available", return_value=False
    ):
        with pytest.raises(RuntimeError) as exc_info:
            tool.run_code_in_sandlock("print('hello')", [])
        assert "Sandlock is not available" in str(exc_info.value)


def test_sandlock_run_success(printer_mock):
    """Test sandlock execution with successful output."""
    tool = CodeInterpreterTool()
    code = "print('hello from sandlock')"

    sandbox_result = SimpleNamespace(
        stdout="hello from sandlock\n", stderr="", returncode=0
    )
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.run.return_value = sandbox_result
    mock_sandbox_cls = MagicMock(return_value=mock_sandbox_instance)
    mock_policy_cls = MagicMock()

    with patch.object(
        CodeInterpreterTool, "_check_sandlock_available", return_value=True
    ):
        mock_sandlock_module = MagicMock()
        mock_sandlock_module.Sandbox = mock_sandbox_cls
        mock_sandlock_module.Policy = mock_policy_cls
        with patch.dict("sys.modules", {"sandlock": mock_sandlock_module}):
            result = tool.run_code_in_sandlock(code, [])

    assert result == "hello from sandlock\n"


def test_sandlock_run_with_error(printer_mock):
    """Test sandlock execution when the code returns an error."""
    tool = CodeInterpreterTool()
    code = "print(1/0)"

    sandbox_result = SimpleNamespace(
        stdout="", stderr="ZeroDivisionError: division by zero", returncode=1
    )
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.run.return_value = sandbox_result
    mock_sandbox_cls = MagicMock(return_value=mock_sandbox_instance)
    mock_policy_cls = MagicMock()

    with patch.object(
        CodeInterpreterTool, "_check_sandlock_available", return_value=True
    ):
        mock_sandlock_module = MagicMock()
        mock_sandlock_module.Sandbox = mock_sandbox_cls
        mock_sandlock_module.Policy = mock_policy_cls
        with patch.dict("sys.modules", {"sandlock": mock_sandlock_module}):
            result = tool.run_code_in_sandlock(code, [])

    assert "Something went wrong" in result
    assert "ZeroDivisionError" in result


def test_sandlock_run_with_exception(printer_mock):
    """Test sandlock execution when an exception occurs."""
    tool = CodeInterpreterTool()
    code = "print('hello')"

    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.run.side_effect = OSError("Landlock not supported")
    mock_sandbox_cls = MagicMock(return_value=mock_sandbox_instance)
    mock_policy_cls = MagicMock()

    with patch.object(
        CodeInterpreterTool, "_check_sandlock_available", return_value=True
    ):
        mock_sandlock_module = MagicMock()
        mock_sandlock_module.Sandbox = mock_sandbox_cls
        mock_sandlock_module.Policy = mock_policy_cls
        with patch.dict("sys.modules", {"sandlock": mock_sandlock_module}):
            result = tool.run_code_in_sandlock(code, [])

    assert "An error occurred in sandlock sandbox" in result
    assert "Landlock not supported" in result


@patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.subprocess.run")
def test_sandlock_installs_libraries_to_temp_dir(
    subprocess_run_mock, printer_mock
):
    """Test that sandlock installs libraries to a temporary directory."""
    tool = CodeInterpreterTool()
    code = "result = 1"
    libraries_used = ["numpy"]

    sandbox_result = SimpleNamespace(stdout="", stderr="", returncode=0)
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.run.return_value = sandbox_result
    mock_sandbox_cls = MagicMock(return_value=mock_sandbox_instance)
    mock_policy_cls = MagicMock()

    with patch.object(
        CodeInterpreterTool, "_check_sandlock_available", return_value=True
    ):
        mock_sandlock_module = MagicMock()
        mock_sandlock_module.Sandbox = mock_sandbox_cls
        mock_sandlock_module.Policy = mock_policy_cls
        with patch.dict("sys.modules", {"sandlock": mock_sandlock_module}):
            tool.run_code_in_sandlock(code, libraries_used)

    # Check that subprocess.run was called for pip install with --target
    pip_calls = [
        c for c in subprocess_run_mock.call_args_list
        if "--target" in c[0][0]
    ]
    assert len(pip_calls) == 1
    args = pip_calls[0][0][0]
    assert args[0] == sys.executable
    assert "--target" in args
    assert "numpy" in args


def test_sandlock_custom_policy_params(printer_mock):
    """Test that custom sandbox parameters are passed to the policy."""
    tool = CodeInterpreterTool(
        sandbox_fs_read=["/custom/read"],
        sandbox_fs_write=["/custom/write"],
        sandbox_max_memory_mb=256,
        sandbox_max_processes=5,
    )

    mock_policy_cls = MagicMock()
    mock_sandlock_module = MagicMock()
    mock_sandlock_module.Policy = mock_policy_cls

    with patch.dict("sys.modules", {"sandlock": mock_sandlock_module}):
        tool._build_sandlock_policy("/tmp/work")

    mock_policy_cls.assert_called_once()
    call_kwargs = mock_policy_cls.call_args[1]
    assert "/custom/read" in call_kwargs["fs_readable"]
    assert "/custom/write" in call_kwargs["fs_writable"]
    assert "/tmp/work" in call_kwargs["fs_writable"]
    assert call_kwargs["max_memory"] == "256M"
    assert call_kwargs["max_processes"] == 5
    assert call_kwargs["isolate_ipc"] is True
    assert call_kwargs["clean_env"] is True


def test_sandlock_default_policy_no_memory_limit(printer_mock):
    """Test that default policy omits max_memory when not configured."""
    tool = CodeInterpreterTool()

    mock_policy_cls = MagicMock()
    mock_sandlock_module = MagicMock()
    mock_sandlock_module.Policy = mock_policy_cls

    with patch.dict("sys.modules", {"sandlock": mock_sandlock_module}):
        tool._build_sandlock_policy("/tmp/work")

    call_kwargs = mock_policy_cls.call_args[1]
    assert "max_memory" not in call_kwargs
    assert "max_processes" not in call_kwargs


def test_sandlock_timeout_default(printer_mock):
    """Test that sandlock uses the default 60s timeout."""
    tool = CodeInterpreterTool()
    code = "print('hello')"

    sandbox_result = SimpleNamespace(stdout="hello\n", stderr="", returncode=0)
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.run.return_value = sandbox_result
    mock_sandbox_cls = MagicMock(return_value=mock_sandbox_instance)
    mock_policy_cls = MagicMock()

    with patch.object(
        CodeInterpreterTool, "_check_sandlock_available", return_value=True
    ):
        mock_sandlock_module = MagicMock()
        mock_sandlock_module.Sandbox = mock_sandbox_cls
        mock_sandlock_module.Policy = mock_policy_cls
        with patch.dict("sys.modules", {"sandlock": mock_sandlock_module}):
            tool.run_code_in_sandlock(code, [])

    # Verify timeout=60 was passed
    run_call = mock_sandbox_instance.run
    assert run_call.call_args[1]["timeout"] == 60


def test_sandlock_custom_timeout(printer_mock):
    """Test that sandlock uses a custom timeout when configured."""
    tool = CodeInterpreterTool(sandbox_timeout=30)
    code = "print('hello')"

    sandbox_result = SimpleNamespace(stdout="hello\n", stderr="", returncode=0)
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.run.return_value = sandbox_result
    mock_sandbox_cls = MagicMock(return_value=mock_sandbox_instance)
    mock_policy_cls = MagicMock()

    with patch.object(
        CodeInterpreterTool, "_check_sandlock_available", return_value=True
    ):
        mock_sandlock_module = MagicMock()
        mock_sandlock_module.Sandbox = mock_sandbox_cls
        mock_sandlock_module.Policy = mock_policy_cls
        with patch.dict("sys.modules", {"sandlock": mock_sandlock_module}):
            tool.run_code_in_sandlock(code, [])

    run_call = mock_sandbox_instance.run
    assert run_call.call_args[1]["timeout"] == 30


def test_auto_mode_prefers_docker_over_sandlock(printer_mock):
    """Test that auto mode tries Docker first before sandlock."""
    tool = CodeInterpreterTool()
    code = "print('hello')"

    with patch.object(
        CodeInterpreterTool, "_check_docker_available", return_value=True
    ):
        with patch.object(
            CodeInterpreterTool, "run_code_in_docker", return_value="hello\n"
        ) as mock_docker:
            with patch.object(
                CodeInterpreterTool,
                "run_code_in_sandlock",
                return_value="hello\n",
            ) as mock_sandlock:
                result = tool.run(code=code, libraries_used=[])

    mock_docker.assert_called_once()
    mock_sandlock.assert_not_called()
    assert result == "hello\n"


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
