import asyncio
import builtins
import importlib
import threading
from unittest.mock import create_autospec

from crewai_tools import SpritesExecTool
from crewai_tools.tools import SpritesExecTool as ExportedSpritesExecTool
from crewai_tools.tools.sprites_tool.sprites_exec_tool import SpritesExecToolSchema
from pydantic import ValidationError
import pytest


@pytest.fixture
def sdk(monkeypatch):
    sprites = pytest.importorskip("sprites")
    from sprites.exec import CompletedProcess

    factory = create_autospec(sprites.SpritesClient)
    client = factory.return_value.__enter__.return_value
    sprite = create_autospec(sprites.Sprite, instance=True)
    client.sprite.return_value = sprite
    sprite.run.return_value = CompletedProcess([], 0, b"hello\n", b"")
    monkeypatch.setattr(sprites, "SpritesClient", factory)
    return factory, client, sprite


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.setenv("SPRITE_TOKEN", "test-sprites-token")
    return SpritesExecTool(sprite_name="crew-workspace")


def test_public_exports():
    assert SpritesExecTool is ExportedSpritesExecTool


def test_command_output_and_existing_sprite_lifecycle(tool, sdk):
    factory, client, sprite = sdk
    result = tool.run(command="printf 'hello\\n'", cwd="/workspace with spaces")
    assert result == {
        "exit_code": 0,
        "stdout": "hello\n",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    factory.assert_called_once_with(token="test-sprites-token")
    client.sprite.assert_called_once_with("crew-workspace")
    sprite.run.assert_called_once_with(
        "bash",
        "-lc",
        "printf 'hello\\n'",
        cwd="/workspace with spaces",
        capture_output=True,
        timeout=60,
        check=False,
    )
    factory.return_value.__exit__.assert_called_once()
    client.create_sprite.assert_not_called()
    client.destroy_sprite.assert_not_called()


def test_nonzero_exit_and_binary_output(tool, sdk):
    from sprites.exec import CompletedProcess

    sdk[2].run.return_value = CompletedProcess([], 7, b"\xff\n", b"failed\n")
    result = tool.run(command="exit 7")
    assert result["exit_code"] == 7
    assert result["stdout"] == "\ufffd\n"
    assert result["stderr"] == "failed\n"


@pytest.mark.parametrize(
    "stdout,stderr,truncated", [(b"abcdef", b"123456", True), (b"abc", b"123", False)]
)
def test_output_limits(tool, sdk, stdout, stderr, truncated):
    from sprites.exec import CompletedProcess

    tool.max_output_chars = 3
    sdk[2].run.return_value = CompletedProcess([], 0, stdout, stderr)
    result = tool.run(command="generate-output")
    assert result["stdout"] == "abc"
    assert result["stderr"] == "123"
    assert result["stdout_truncated"] is truncated
    assert result["stderr_truncated"] is truncated


def test_empty_output(tool, sdk):
    from sprites.exec import CompletedProcess

    sdk[2].run.return_value = CompletedProcess([], 0)
    result = tool.run(command="true")
    assert result["stdout"] == result["stderr"] == ""


def test_credentials_are_not_agent_arguments_or_serialized(tool, sdk):
    assert tool.args_schema is SpritesExecToolSchema
    assert set(tool.args_schema.model_json_schema()["properties"]) == {"command", "cwd"}
    assert "api_key" not in tool.model_dump()
    assert "test-sprites-token" not in tool.model_dump_json()
    assert "test-sprites-token" not in repr(tool)
    assert "test-sprites-token" not in tool.description


def test_explicit_token_overrides_environment(sdk, monkeypatch):
    monkeypatch.setenv("SPRITE_TOKEN", "environment-token")
    tool = SpritesExecTool(
        sprite_name="crew-workspace", api_key="explicit-token", timeout=15
    )
    tool.run(command="true")
    sdk[0].assert_called_once_with(token="explicit-token")
    assert sdk[2].run.call_args.kwargs["timeout"] == 15


@pytest.mark.parametrize("token", [None, "", " "])
def test_missing_credentials(token, monkeypatch, sdk):
    monkeypatch.delenv("SPRITE_TOKEN", raising=False)
    tool = SpritesExecTool(sprite_name="crew-workspace", api_key=token)
    with pytest.raises(ValueError, match="SPRITE_TOKEN"):
        tool.run(command="true")
    sdk[0].assert_not_called()


def test_optional_sdk_is_lazy_and_has_install_hint(tool, monkeypatch):
    original_import = builtins.__import__

    def without_sprites(name, *args, **kwargs):
        if name == "sprites" or name.startswith("sprites."):
            raise ImportError("No module named sprites")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sprites)
    importlib.reload(importlib.import_module("crewai_tools"))
    with pytest.raises(ImportError, match=r"crewai-tools\[sprites\]"):
        tool.run(command="true")


@pytest.mark.parametrize("sdk_timeout", [False, True])
def test_timeout_warns_about_unknown_remote_state_and_closes_client(
    tool, sdk, sdk_timeout
):
    from sprites.exceptions import TimeoutError as SpritesTimeoutError

    sdk[2].run.side_effect = (SpritesTimeoutError if sdk_timeout else TimeoutError)(
        "private request"
    )
    with pytest.raises(TimeoutError, match="may still be running"):
        tool.run(command="long-running-command")
    sdk[0].return_value.__exit__.assert_called_once()
    sdk[2].run.assert_called_once()  # Never automatically retry a side effect.
    sdk[1].destroy_sprite.assert_not_called()


def test_sdk_failure_does_not_expose_credentials_or_retry(tool, sdk):
    sdk[2].run.side_effect = RuntimeError("Authorization: Bearer test-sprites-token")
    with pytest.raises(RuntimeError, match="outcome may be unknown") as error:
        tool.run(command="write-file")
    assert "test-sprites-token" not in str(error.value)
    assert error.value.__suppress_context__
    sdk[0].return_value.__exit__.assert_called_once()
    sdk[2].run.assert_called_once()


@pytest.mark.parametrize(
    "name", ["", " ", "../other", ".", "..", "a?b", "a#b", "a%2fb", "a\\b", "a\x00b"]
)
def test_invalid_sprite_names(name):
    with pytest.raises(ValidationError):
        SpritesExecTool(sprite_name=name)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": 301},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"max_output_chars": 0},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(ValidationError):
        SpritesExecTool(sprite_name="crew-workspace", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command": ""},
        {"command": " "},
        {"command": "a\x00b"},
        {"command": "true", "cwd": " "},
        {"command": "true", "cwd": "a\x00b"},
    ],
)
def test_invalid_arguments(tool, sdk, kwargs):
    with pytest.raises(ValueError):
        tool.run(**kwargs)
    sdk[0].assert_not_called()


def test_commands_are_not_cacheable(tool):
    assert tool.cache_function({"command": "true"}, {"exit_code": 0}) is False
    assert tool.to_structured_tool().cache_function({}, {}) is False


@pytest.mark.asyncio
async def test_async_execution_uses_worker_thread(tool, sdk):
    main_thread = threading.get_ident()
    threads = []
    result = sdk[2].run.return_value

    def run(*args, **kwargs):
        threads.append(threading.get_ident())
        return result

    sdk[2].run.side_effect = run
    outputs = await asyncio.gather(tool.arun(command="true"), tool.arun(command="true"))
    assert all(output["exit_code"] == 0 for output in outputs)
    assert len(threads) == 2
    assert all(thread != main_thread for thread in threads)
    assert sdk[0].return_value.__exit__.call_count == 2


def test_real_sdk_command_contract(monkeypatch, tool):
    """Exercise the installed SDK and CrewAI adapter, replacing only network I/O."""
    pytest.importorskip("sprites")
    from sprites.exec import Cmd

    def run_without_network(command):
        assert command.args == ["bash", "-lc", "printf hello"]
        assert command.sprite.name == "crew-workspace"
        assert command.dir == "/workspace"
        assert command.env == {}  # The host token is never injected remotely.
        assert command.timeout == 60
        assert command._capture_stdout and command._capture_stderr
        command._stdout_data = b"hello"
        command._stderr_data = b"warning"
        return 3

    monkeypatch.setattr(Cmd, "_run_sync", run_without_network)
    result = tool.to_structured_tool().invoke(
        {"command": "printf hello", "cwd": "/workspace"}
    )
    assert result["exit_code"] == 3
    assert result["stdout"] == "hello"
    assert result["stderr"] == "warning"
