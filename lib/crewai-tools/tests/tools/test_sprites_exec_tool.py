import asyncio
import builtins
from collections.abc import AsyncIterator, Iterable
import importlib
from typing import Any
from unittest.mock import AsyncMock, create_autospec

from crewai_tools import SpritesExecTool
from crewai_tools.tools import SpritesExecTool as ExportedSpritesExecTool
from crewai_tools.tools.sprites_tool.sprites_exec_tool import SpritesExecToolSchema
from pydantic import ValidationError
import pytest


class FakeSocket:
    """Replace network I/O while exercising the installed SDK's receive loop."""

    def __init__(self) -> None:
        """Provide a successful command by default, with optional blocking I/O."""
        self.messages: Iterable[str | bytes] = [b"\x01hello\n", b"\x03\x00"]
        self.block = False
        self.reading = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close = AsyncMock()
        self.send = AsyncMock()
        self.close_code = 1000
        self.close_reason = ""

    async def __aiter__(self) -> AsyncIterator[str | bytes]:
        """Yield frames, or await cancellation like an idle WebSocket."""
        self.reading = True
        self.started.set()
        try:
            if self.block:
                await self.release.wait()
            for message in self.messages:
                yield message
                await asyncio.sleep(0)
        finally:
            self.reading = False


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any, FakeSocket]:
    """Mock client construction and the socket, retaining real SDK commands."""
    sprites = pytest.importorskip("sprites")
    from sprites import websocket
    from sprites.exec import Cmd

    factory = create_autospec(sprites.SpritesClient)
    client = factory.return_value.__enter__.return_value
    client.base_url = "https://api.sprites.dev"
    client.token = "test-sprites-token"
    sprite = create_autospec(sprites.Sprite, instance=True)
    sprite.name = "crew-workspace"
    sprite.client = client
    sprite.command.side_effect = lambda *args, **kwargs: Cmd(
        sprite, list(args), **kwargs
    )
    client.sprite.return_value = sprite
    socket = FakeSocket()
    monkeypatch.setattr(websocket, "connect", AsyncMock(return_value=socket))
    monkeypatch.setattr(sprites, "SpritesClient", factory)
    return factory, client, sprite, socket


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch) -> SpritesExecTool:
    """Construct a tool with a deliberately fake environment token."""
    monkeypatch.setenv("SPRITE_TOKEN", "test-sprites-token")
    return SpritesExecTool(sprite_name="crew-workspace")


def test_public_exports() -> None:
    """Expose the tool consistently through both public import paths."""
    assert SpritesExecTool is ExportedSpritesExecTool


def test_command_output_and_existing_sprite_lifecycle(
    tool: SpritesExecTool, sdk: Any
) -> None:
    """Preserve shell arguments, return output, and never change Sprite lifecycle."""
    factory, client, sprite, socket = sdk
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
    sprite.command.assert_called_once_with(
        "bash",
        "-lc",
        "printf 'hello\\n'",
        cwd="/workspace with spaces",
        timeout=60,
    )
    socket.close.assert_awaited_once()
    socket.send.assert_awaited_once_with(b"\x04")
    factory.return_value.__exit__.assert_called_once()
    client.create_sprite.assert_not_called()
    client.destroy_sprite.assert_not_called()


def test_nonzero_exit_and_binary_output(tool: SpritesExecTool, sdk: Any) -> None:
    """Return nonzero exits and replace invalid UTF-8 without hiding stderr."""
    sdk[3].messages = [b"\x01\xff\n", b"\x02failed\n", b"\x03\x07"]
    result = tool.run(command="exit 7")
    assert result["exit_code"] == 7
    assert result["stdout"] == "\ufffd\n"
    assert result["stderr"] == "failed\n"


@pytest.mark.parametrize(
    "stdout,stderr,truncated", [(b"abcdef", b"123456", True), (b"abc", b"123", False)]
)
def test_output_limits(
    tool: SpritesExecTool, sdk: Any, stdout: bytes, stderr: bytes, truncated: bool
) -> None:
    """Report truncation only when a stream exceeds its character limit."""
    tool.max_output_chars = 3
    sdk[3].messages = [b"\x01" + stdout, b"\x02" + stderr, b"\x03\x00"]
    result = tool.run(command="generate-output")
    assert result["stdout"] == "abc"
    assert result["stderr"] == "123"
    assert result["stdout_truncated"] is truncated
    assert result["stderr_truncated"] is truncated


def test_empty_output(tool: SpritesExecTool, sdk: Any) -> None:
    """Represent absent stdout and stderr as empty strings."""
    sdk[3].messages = [b"\x03\x00"]
    result = tool.run(command="true")
    assert result["stdout"] == result["stderr"] == ""


def test_credentials_are_not_agent_arguments_or_serialized(
    tool: SpritesExecTool, sdk: Any
) -> None:
    """Keep credentials out of the agent schema and serialized tool state."""
    assert tool.args_schema is SpritesExecToolSchema
    assert set(tool.args_schema.model_json_schema()["properties"]) == {"command", "cwd"}
    assert "api_key" not in tool.model_dump()
    assert "test-sprites-token" not in tool.model_dump_json()
    assert "test-sprites-token" not in repr(tool)
    assert "test-sprites-token" not in tool.description


def test_explicit_token_overrides_environment(
    sdk: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefer explicit credentials and pass through the configured timeout."""
    monkeypatch.setenv("SPRITE_TOKEN", "environment-token")
    tool = SpritesExecTool(
        sprite_name="crew-workspace", api_key="explicit-token", timeout=15
    )
    tool.run(command="true")
    sdk[0].assert_called_once_with(token="explicit-token")
    assert sdk[2].command.call_args.kwargs["timeout"] == 15


@pytest.mark.parametrize("token", [None, "", " "])
def test_missing_credentials(
    token: str | None, monkeypatch: pytest.MonkeyPatch, sdk: Any
) -> None:
    """Reject missing or blank credentials before opening a client."""
    monkeypatch.delenv("SPRITE_TOKEN", raising=False)
    tool = SpritesExecTool(sprite_name="crew-workspace", api_key=token)
    with pytest.raises(ValueError, match="SPRITE_TOKEN"):
        tool.run(command="true")
    sdk[0].assert_not_called()


def test_optional_sdk_is_lazy_and_has_install_hint(
    tool: SpritesExecTool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow importing CrewAI tools without the optional SDK installed."""
    original_import = builtins.__import__

    def without_sprites(name: str, *args: Any, **kwargs: Any) -> Any:
        """Simulate an environment without any Sprites SDK modules."""
        if name == "sprites" or name.startswith("sprites."):
            raise ImportError("No module named sprites")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sprites)
    importlib.reload(importlib.import_module("crewai_tools"))
    with pytest.raises(ImportError, match=r"crewai-tools\[sprites\]"):
        tool.run(command="true")


@pytest.mark.parametrize("sdk_timeout", [False, True])
def test_timeout_warns_about_unknown_remote_state_and_closes_client(
    tool: SpritesExecTool, sdk: Any, sdk_timeout: bool
) -> None:
    """Sanitize both timeout types without retrying a remote side effect."""
    from sprites.exceptions import TimeoutError as SpritesTimeoutError

    sdk[2].command.side_effect = (SpritesTimeoutError if sdk_timeout else TimeoutError)(
        "private request"
    )
    with pytest.raises(TimeoutError, match="may still be running"):
        tool.run(command="long-running-command")
    sdk[0].return_value.__exit__.assert_called_once()
    sdk[2].command.assert_called_once()  # Never automatically retry a side effect.
    sdk[1].destroy_sprite.assert_not_called()


def test_sdk_failure_does_not_expose_credentials_or_retry(
    tool: SpritesExecTool, sdk: Any
) -> None:
    """Suppress credential-bearing SDK errors and leave remote state alone."""
    sdk[2].command.side_effect = RuntimeError(
        "Authorization: Bearer test-sprites-token"
    )
    with pytest.raises(RuntimeError, match="outcome may be unknown") as error:
        tool.run(command="write-file")
    assert "test-sprites-token" not in str(error.value)
    assert error.value.__suppress_context__
    sdk[0].return_value.__exit__.assert_called_once()
    sdk[2].command.assert_called_once()


@pytest.mark.parametrize(
    "name", ["", " ", "../other", ".", "..", "a?b", "a#b", "a%2fb", "a\\b", "a\x00b"]
)
def test_invalid_sprite_names(name: str) -> None:
    """Disallow invalid names and path/query injection through the Sprite name."""
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
def test_invalid_limits(kwargs: dict[str, Any]) -> None:
    """Reject nonpositive limits and timeouts beyond the documented bound."""
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
def test_invalid_arguments(
    tool: SpritesExecTool, sdk: Any, kwargs: dict[str, str]
) -> None:
    """Validate command arguments before SDK client construction."""
    with pytest.raises(ValueError):
        tool.run(**kwargs)
    sdk[0].assert_not_called()


def test_commands_are_not_cacheable(tool: SpritesExecTool) -> None:
    """Prevent cached results from skipping side-effecting commands."""
    assert tool.cache_function({"command": "true"}, {"exit_code": 0}) is False
    assert tool.to_structured_tool().cache_function({}, {}) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_async_execution_avoids_executor(
    tool: SpritesExecTool, sdk: Any, monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    """Run concurrent async commands without submitting any executor work."""
    submit = create_autospec(asyncio.get_running_loop().run_in_executor)
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", submit)
    adapter = tool.to_structured_tool()
    outputs = await asyncio.gather(
        *(
            adapter.ainvoke({"command": "true"})
            if structured
            else tool.arun(command="true")
            for _ in range(2)
        )
    )
    assert all(output["exit_code"] == 0 for output in outputs)
    submit.assert_not_called()
    assert sdk[0].return_value.__exit__.call_count == 2


def test_real_sdk_command_contract(
    monkeypatch: pytest.MonkeyPatch, tool: SpritesExecTool
) -> None:
    """Exercise the real client, SDK transport and CrewAI adapter without network."""
    from urllib.parse import parse_qs, urlsplit

    from sprites import websocket

    socket = FakeSocket()
    socket.messages = [b"\x01hello", b"\x02warning", b"\x03\x03"]
    connect = AsyncMock(return_value=socket)
    monkeypatch.setattr(websocket, "connect", connect)
    result = tool.to_structured_tool().invoke(
        {"command": "printf hello", "cwd": "/workspace"}
    )
    assert result["exit_code"] == 3
    assert result["stdout"] == "hello"
    assert result["stderr"] == "warning"
    url = urlsplit(connect.call_args.args[0])
    assert url.scheme == "wss"
    assert url.path == "/v1/sprites/crew-workspace/exec"
    query = parse_qs(url.query)
    assert query["cmd"] == ["bash", "-lc", "printf hello"]
    assert query["dir"] == ["/workspace"]
    assert "env" not in query  # Never inject the host token into remote commands.
    assert connect.call_args.kwargs["additional_headers"]["Authorization"] == (
        "Bearer test-sprites-token"
    )
    socket.close.assert_awaited_once()


@pytest.mark.parametrize("frames", [[b"abc"], [b"ab", b"c"], [b"abcdef"]])
def test_utf8_prefix_across_frames(
    tool: SpritesExecTool, sdk: Any, frames: list[bytes]
) -> None:
    """Decode split multibyte sequences and count characters, not bytes."""
    tool.max_output_chars = 3
    emoji = "😀".encode()
    sdk[3].messages = [b"\x01" + emoji[:2], b"\x01" + emoji[2:], b"\x01\xc3"]
    sdk[3].messages += (
        [b"\x01\xa9"] + [b"\x01" + part for part in frames] + [b"\x03\x00"]
    )
    result = tool.run(command="unicode-output")
    assert result["stdout"] == "😀éa"
    assert result["stdout_truncated"] is True


@pytest.mark.parametrize(
    "data,expected,truncated",
    [(b"ab\xe2", "ab�", False), (b"abc\xe2", "abc", True), (b"abc", "abc", False)],
)
def test_incomplete_utf8_at_exit(
    tool: SpritesExecTool, sdk: Any, data: bytes, expected: str, truncated: bool
) -> None:
    """Flush an incomplete final code point before computing truncation flags."""
    tool.max_output_chars = 3
    sdk[3].messages = [b"\x01" + data, b"\x03\x00"]
    result = tool.run(command="binary-output")
    assert result["stdout"] == expected
    assert result["stdout_truncated"] is truncated


@pytest.mark.asyncio
async def test_large_output_never_accumulates_in_sdk(sdk: Any) -> None:
    """Drain many frames past the cap without growing either SDK output buffer."""
    from crewai_tools.tools.sprites_tool._execution import BoundedWSCommand

    command = sdk[2].command("bash", "-lc", "noisy-command")
    execution = BoundedWSCommand(command, 3)

    def frames() -> Iterable[bytes]:
        """Generate megabytes without preallocating a complete result."""
        for _ in range(512):
            yield b"\x01" + b"a" * 8192
            yield b"\x02" + b"b" * 8192
            assert execution.stdout.length <= 3
            assert execution.stderr.length <= 3
            assert execution.get_stdout() == execution.get_stderr() == b""
            assert sum(map(len, execution.stdout.parts)) <= 3
            assert sum(map(len, execution.stderr.parts)) <= 3
        yield b"\x03\x07"

    sdk[3].messages = frames()
    assert await execution.execute() == 7
    assert execution.stdout.text() == "aaa"
    assert execution.stderr.text() == "bbb"
    assert execution.stdout.truncated and execution.stderr.truncated
    sdk[3].close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_cancellation_closes_connection_without_executor(
    tool: SpritesExecTool, sdk: Any, monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    """Repeated cancellations leave neither shared workers nor receive tasks."""
    from sprites import websocket

    submit = create_autospec(asyncio.get_running_loop().run_in_executor)
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", submit)
    baseline = asyncio.all_tasks()
    for _ in range(5):
        socket = FakeSocket()
        socket.block = True
        monkeypatch.setattr(websocket, "connect", AsyncMock(return_value=socket))
        task = asyncio.create_task(
            tool.to_structured_tool().ainvoke({"command": "long-command"})
            if structured
            else tool.arun(command="long-command")
        )
        await asyncio.wait_for(socket.started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not socket.reading
        socket.close.assert_awaited_once()
    assert asyncio.all_tasks() == baseline
    submit.assert_not_called()
    assert sdk[0].return_value.__exit__.call_count == 5
    sdk[1].destroy_sprite.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_cancels_receive_and_closes_connection(
    tool: SpritesExecTool, sdk: Any
) -> None:
    """Enforce the configured deadline while cleaning up local I/O."""
    tool.timeout = 0.02
    sdk[3].block = True
    with pytest.raises(TimeoutError, match="may still be running"):
        await tool.arun(command="long-command")
    assert not sdk[3].reading
    sdk[3].close.assert_awaited_once()
    sdk[0].return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("cancel", [True, False])
@pytest.mark.asyncio
async def test_interrupt_during_connection_setup(
    tool: SpritesExecTool, sdk: Any, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    """Cancel or time out a pending connection without leaving a connect task."""
    from sprites import websocket

    started = asyncio.Event()
    finished = asyncio.Event()

    async def connect(*args: Any, **kwargs: Any) -> None:
        """Remain in connection setup until the task is cancelled."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    monkeypatch.setattr(websocket, "connect", connect)
    tool.timeout = 1 if cancel else 0.02
    task = asyncio.create_task(tool.arun(command="true"))
    await asyncio.wait_for(started.wait(), timeout=1)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert finished.is_set()
    sdk[0].return_value.__exit__.assert_called_once()


def test_disconnect_before_exit_is_not_success(tool: SpritesExecTool, sdk: Any) -> None:
    """Treat a transport close without an exit frame as an unknown outcome."""
    sdk[3].messages = [b"\x01partial"]
    with pytest.raises(RuntimeError, match="outcome may be unknown"):
        tool.run(command="write-file")
    sdk[3].close.assert_awaited_once()


def test_close_failure_preserves_command_exit(tool: SpritesExecTool, sdk: Any) -> None:
    """Keep a known command result even if the close handshake fails."""
    sdk[3].close.side_effect = RuntimeError("close failed")
    assert tool.run(command="true")["exit_code"] == 0


@pytest.mark.parametrize("limit", [1, 3, 20])
@pytest.mark.parametrize("chunk_size", [1, 2, 7, 1024])
def test_bounded_collector_matches_utf8_decode(limit: int, chunk_size: int) -> None:
    """Match full UTF-8 decoding for valid, invalid, and split byte sequences."""
    from crewai_tools.tools.sprites_tool._execution import _BoundedOutput

    for data in (
        b"",
        "aé😀日".encode() * 50,
        bytes(range(256)) * 5,
        b"\xf0\x9f\x98",
        b"abc\xf0\x9f\x98",
        b"\xed\xa0\x80\xff\xc0\xaf" * 20,
    ):
        output = _BoundedOutput(limit)
        for start in range(0, len(data), chunk_size):
            output.write(data[start : start + chunk_size])
            assert output.length <= limit
            assert sum(map(len, output.parts)) <= limit
        expected = data.decode("utf-8", errors="replace")
        assert output.text() == expected[:limit]
        assert output.truncated is (len(expected) > limit)


def test_control_frames_and_text_exit(tool: SpritesExecTool, sdk: Any) -> None:
    """Leave session metadata and text exit handling with the real SDK."""
    sdk[3].messages = [
        '{"type":"session_info","tty":false}',
        b"",
        b"\x01hello",
        '{"type":"exit","exit_code":9}',
    ]
    result = tool.run(command="true")
    assert result["stdout"] == "hello"
    assert result["exit_code"] == 9


def test_unexpected_tty_does_not_fall_back_to_unbounded_capture(
    tool: SpritesExecTool, sdk: Any
) -> None:
    """Fail closed if the server unexpectedly switches to a TTY protocol."""
    sdk[3].messages = ['{"type":"session_info","tty":true}', b"raw-output"]
    with pytest.raises(RuntimeError, match="outcome may be unknown"):
        tool.run(command="true")
    sdk[3].close.assert_awaited_once()
