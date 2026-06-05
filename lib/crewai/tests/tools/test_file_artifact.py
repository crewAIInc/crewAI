"""Tests for out-of-band binary file passing between tools."""

from __future__ import annotations

import base64
import re

import pytest

from crewai.tools import FileArtifact
from crewai.tools.file_artifact import (
    _store,
    artifact_scope_id,
    clear_artifact_scope,
    resolve_artifact_handles,
    store_artifact,
    store_if_artifact,
)


_HANDLE = re.compile(r"crewai\+file://[0-9a-fA-F-]{36}")


@pytest.fixture(autouse=True)
def _clear_store():
    """Keep the process-local store empty between tests."""
    _store._entries.clear()
    yield
    _store._entries.clear()


def _handle_in(text: str) -> str:
    match = _HANDLE.search(text)
    assert match is not None, f"no handle in: {text!r}"
    return match.group(0)


class TestFileArtifact:
    def test_as_base64_round_trips(self) -> None:
        data = bytes(range(256))
        artifact = FileArtifact(data=data, filename="x.bin")
        assert base64.b64decode(artifact.as_base64()) == data

    def test_size_bytes(self) -> None:
        assert FileArtifact(data=b"abc").size_bytes == 3

    def test_defaults(self) -> None:
        artifact = FileArtifact(data=b"")
        assert artifact.filename == "file"
        assert artifact.mime_type == "application/octet-stream"


class TestStoreArtifact:
    def test_placeholder_contains_metadata_and_handle(self) -> None:
        artifact = FileArtifact(
            data=b"\x00" * 30045, filename="deck.pptx", mime_type="application/pptx"
        )
        placeholder = store_artifact(artifact, scope_id="crew-1")
        assert 'filename="deck.pptx"' in placeholder
        assert 'mime_type="application/pptx"' in placeholder
        assert "29.3 KB" in placeholder
        assert _HANDLE.search(placeholder) is not None

    def test_each_store_gets_a_unique_handle(self) -> None:
        h1 = _handle_in(store_artifact(FileArtifact(data=b"a")))
        h2 = _handle_in(store_artifact(FileArtifact(data=b"a")))
        assert h1 != h2

    def test_placeholder_escapes_quotes_in_metadata(self) -> None:
        artifact = FileArtifact(data=b"x", filename='a".pptx', mime_type='m"/x')
        placeholder = store_artifact(artifact)
        # The bracketed attribute list must not be broken by an embedded quote,
        # and the handle must still be recoverable.
        assert 'filename="a\'.pptx"' in placeholder
        assert _HANDLE.search(placeholder) is not None


class TestArtifactScopeId:
    class _Obj:
        def __init__(self, id_):
            self.id = id_

    def test_prefers_crew_id(self) -> None:
        assert artifact_scope_id(self._Obj("crew"), self._Obj("task")) == "crew"

    def test_falls_back_to_task_when_no_crew(self) -> None:
        assert artifact_scope_id(None, self._Obj("task")) == "task"

    def test_falls_back_to_task_when_crew_id_is_none(self) -> None:
        assert artifact_scope_id(self._Obj(None), self._Obj("task")) == "task"

    def test_none_when_neither_present(self) -> None:
        assert artifact_scope_id(None, None) is None


class TestResolveArtifactHandles:
    def test_exact_handle_resolves_to_base64(self) -> None:
        data = bytes(range(256)) * 100
        handle = _handle_in(store_artifact(FileArtifact(data=data)))
        resolved = resolve_artifact_handles(handle)
        assert base64.b64decode(resolved) == data

    def test_resolves_handle_inside_dict(self) -> None:
        data = b"binary-payload" * 1000
        handle = _handle_in(store_artifact(FileArtifact(data=data)))
        args = {"file_name": "a.bin", "content": handle}
        resolved = resolve_artifact_handles(args)
        assert base64.b64decode(resolved["content"]) == data
        assert resolved["file_name"] == "a.bin"

    def test_resolves_handle_nested_in_list_and_dict(self) -> None:
        handle = _handle_in(store_artifact(FileArtifact(data=b"xyz")))
        resolved = resolve_artifact_handles({"items": [{"c": handle}]})
        assert base64.b64decode(resolved["items"][0]["c"]) == b"xyz"

    def test_does_not_mutate_original_arguments(self) -> None:
        handle = _handle_in(store_artifact(FileArtifact(data=b"data")))
        args = {"content": handle}
        resolve_artifact_handles(args)
        assert args["content"] == handle

    def test_unknown_handle_is_left_unchanged(self) -> None:
        token = "crewai+file://00000000-0000-0000-0000-000000000000"
        assert resolve_artifact_handles(token) == token

    def test_non_handle_strings_pass_through(self) -> None:
        assert resolve_artifact_handles("just text") == "just text"
        assert resolve_artifact_handles({"k": "v"}) == {"k": "v"}

    def test_non_string_values_pass_through(self) -> None:
        assert resolve_artifact_handles(42) == 42
        assert resolve_artifact_handles(None) is None
        assert resolve_artifact_handles([1, 2]) == [1, 2]


class TestStoreIfArtifact:
    def test_artifact_becomes_placeholder(self) -> None:
        result = store_if_artifact(FileArtifact(data=b"a" * 100), scope_id="s")
        assert isinstance(result, str)
        assert _HANDLE.search(result) is not None

    def test_other_values_unchanged(self) -> None:
        assert store_if_artifact("hello") == "hello"
        assert store_if_artifact(7) == 7


class TestScoping:
    def test_clear_scope_only_drops_its_own_artifacts(self) -> None:
        h_a = _handle_in(store_artifact(FileArtifact(data=b"a"), scope_id="A"))
        h_b = _handle_in(store_artifact(FileArtifact(data=b"b"), scope_id="B"))

        clear_artifact_scope("A")

        # A's handle no longer resolves; B's still does.
        assert resolve_artifact_handles(h_a) == h_a
        assert base64.b64decode(resolve_artifact_handles(h_b)) == b"b"

    def test_unscoped_artifact_survives_other_scope_clears(self) -> None:
        handle = _handle_in(store_artifact(FileArtifact(data=b"x")))
        clear_artifact_scope("some-crew")
        assert base64.b64decode(resolve_artifact_handles(handle)) == b"x"


def _legacy_executor_runner(tools):
    """Return a `(func_name, args) -> result_dict` driver for the legacy executor."""
    from unittest.mock import Mock

    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    from crewai.tools.base_tool import to_langchain
    from crewai.utilities.agent_utils import convert_tools_to_openai_schema

    executor = CrewAgentExecutor(tools=to_langchain(tools), original_tools=tools)
    agent = Mock(key="agent", role="tester", verbose=False, fingerprint=None)
    agent.tools_results = []
    executor.agent = agent
    task = Mock(description="t", id="scope-legacy")
    task.name = "t"  # `name=` is a reserved Mock ctor kwarg, so assign explicitly
    executor.task = task
    _, available_functions, _ = convert_tools_to_openai_schema(tools)

    def run(func_name, args):
        return executor._execute_single_native_tool_call(
            call_id="c",
            func_name=func_name,
            func_args=args,
            available_functions=available_functions,
        )

    return run


def _experimental_executor_runner(tools):
    """Return a `(func_name, args) -> result_dict` driver for the default executor."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from crewai.experimental.agent_executor import AgentExecutor

    executor = AgentExecutor.model_construct()
    for key, value in {
        "original_tools": tools,
        "tools": [],
        "tools_handler": None,
        "crew": None,
    }.items():
        object.__setattr__(executor, key, value)
    agent = Mock(key="agent", role="tester", verbose=False, fingerprint=None)
    agent.tools_results = []
    object.__setattr__(executor, "agent", agent)
    task = Mock(id="scope-exp", description="t")
    task.name = "t"  # `name=` is a reserved Mock ctor kwarg, so assign explicitly
    object.__setattr__(executor, "task", task)
    executor._setup_native_tools()

    def run(func_name, args):
        tool_call = SimpleNamespace(
            id="c",
            function=SimpleNamespace(
                name=func_name, arguments=args if isinstance(args, str) else json.dumps(args)
            ),
        )
        return executor._execute_single_native_tool_call(tool_call)

    return run


@pytest.mark.parametrize(
    "make_runner",
    [_experimental_executor_runner, _legacy_executor_runner],
    ids=["experimental", "legacy"],
)
class TestNativeExecutorWiring:
    """Guard producer/consumer wiring on both the default and legacy executors."""

    def test_artifact_output_is_replaced_by_handle_and_resolves_downstream(
        self, make_runner
    ) -> None:
        from crewai.tools import BaseTool, FileArtifact

        payload = bytes(range(256)) * 200  # ~51 KB, far past the LLM round-trip limit

        class Generate(BaseTool):
            name: str = "generate_file"
            description: str = "Generate a binary file"

            def _run(self) -> FileArtifact:
                return FileArtifact(
                    data=payload, filename="deck.pptx", mime_type="application/pptx"
                )

        captured: dict[str, str] = {}

        class Upload(BaseTool):
            name: str = "upload_file"
            description: str = "Upload base64 content"

            def _run(self, content: str) -> str:
                captured["content"] = content
                return "uploaded"

        run = make_runner([Generate(), Upload()])

        # Producer: the 51 KB payload must NOT appear in the model-facing result.
        gen_result = run("generate_file", "{}")["result"]
        assert "deck.pptx" in gen_result
        assert base64.b64encode(payload).decode() not in gen_result
        handle = _handle_in(gen_result)

        # Consumer: the handle the model echoes is expanded to exact bytes.
        up_result = run("upload_file", {"content": handle})["result"]
        assert up_result == "uploaded"
        assert base64.b64decode(captured["content"]) == payload


class TestTtlPrune:
    def test_expired_entries_are_pruned_on_next_store(self) -> None:
        stale = _handle_in(store_artifact(FileArtifact(data=b"old"), ttl=3600))
        stale_id = stale.rsplit("/", 1)[-1]
        # Force the entry to look old, then trigger a prune via another store.
        _store._entries[stale_id].stored_at -= 7200
        store_artifact(FileArtifact(data=b"new"), ttl=3600)
        assert resolve_artifact_handles(stale) == stale

    def test_ttl_zero_disables_pruning(self) -> None:
        handle = _handle_in(store_artifact(FileArtifact(data=b"keep"), ttl=0))
        handle_id = handle.rsplit("/", 1)[-1]
        _store._entries[handle_id].stored_at -= 99999
        store_artifact(FileArtifact(data=b"another"), ttl=0)
        assert base64.b64decode(resolve_artifact_handles(handle)) == b"keep"
