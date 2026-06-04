"""Tests for the ToolFileReference sideband file store system."""

from __future__ import annotations

import base64

import pytest

from crewai.tools.tool_file_reference import (
    ToolFileReference,
    ToolFileStore,
    auto_store_if_binary,
    is_large_base64,
    tool_file_store,
)


class TestToolFileStore:
    def setup_method(self) -> None:
        self.store = ToolFileStore()

    def test_store_and_resolve(self) -> None:
        data = b"hello world"
        ref = self.store.store(data, filename="test.txt", content_type="text/plain")
        assert ref.filename == "test.txt"
        assert ref.content_type == "text/plain"
        assert ref.size_bytes == len(data)
        assert self.store.resolve(ref.ref_id) == data

    def test_resolve_reference(self) -> None:
        data = b"\x00\x01\x02"
        ref = self.store.store(data, filename="bin.dat")
        resolved = self.store.resolve_reference(ref.ref_id)
        assert resolved is ref
        assert resolved.data == data

    def test_resolve_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="File reference not found"):
            self.store.resolve("nonexistent-id")

    def test_resolve_reference_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="File reference not found"):
            self.store.resolve_reference("nonexistent-id")

    def test_has(self) -> None:
        ref = self.store.store(b"data")
        assert self.store.has(ref.ref_id)
        assert not self.store.has("fake-id")

    def test_clear(self) -> None:
        ref = self.store.store(b"data")
        self.store.clear()
        assert not self.store.has(ref.ref_id)

    def test_placeholder_format(self) -> None:
        ref = ToolFileReference(
            ref_id="abc-123",
            filename="report.pdf",
            size_bytes=46080,
            data=b"x" * 46080,
        )
        assert ref.placeholder() == "[File: report.pdf, 45.0 KB, ref=abc-123]"


class TestIsLargeBase64:
    def test_small_string_returns_false(self) -> None:
        assert is_large_base64("aGVsbG8=") is False

    def test_non_base64_returns_false(self) -> None:
        assert is_large_base64("not base64 at all! @#$" * 500) is False

    def test_large_base64_returns_true(self) -> None:
        data = base64.b64encode(b"\x00" * 5000).decode()
        assert is_large_base64(data) is True

    def test_custom_threshold(self) -> None:
        data = base64.b64encode(b"\x00" * 100).decode()
        assert is_large_base64(data, threshold=50) is True
        assert is_large_base64(data, threshold=200) is False


class TestAutoStoreIfBinary:
    def setup_method(self) -> None:
        tool_file_store.clear()

    def test_passthrough_for_short_strings(self) -> None:
        result = auto_store_if_binary("hello world")
        assert result == "hello world"

    def test_passthrough_for_non_string(self) -> None:
        result = auto_store_if_binary(42)
        assert result == 42

    def test_passthrough_for_existing_reference(self) -> None:
        ref = ToolFileReference(filename="x.bin", data=b"data")
        result = auto_store_if_binary(ref)
        assert result is ref

    def test_stores_large_base64(self) -> None:
        raw = b"\x00" * 5000
        b64_str = base64.b64encode(raw).decode()
        result = auto_store_if_binary(b64_str)
        assert isinstance(result, ToolFileReference)
        assert result.size_bytes == 5000
        assert tool_file_store.resolve(result.ref_id) == raw


class TestResolveFileReferences:
    """Test the _resolve_file_references static method on CrewAgentExecutor."""

    def setup_method(self) -> None:
        tool_file_store.clear()

    def test_resolves_ref_id_in_args(self) -> None:
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        data = b"binary content here"
        ref = tool_file_store.store(data, filename="upload.bin")

        args = {"file_content": ref.ref_id, "name": "test.bin"}
        resolved = CrewAgentExecutor._resolve_file_references(args)

        assert resolved is not None
        assert resolved["file_content"] == base64.b64encode(data).decode("ascii")
        assert resolved["name"] == "test.bin"

    def test_passthrough_non_ref_args(self) -> None:
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        args = {"query": "hello", "count": "5"}
        resolved = CrewAgentExecutor._resolve_file_references(args)
        assert resolved == args

    def test_none_args(self) -> None:
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        assert CrewAgentExecutor._resolve_file_references(None) is None

    def test_empty_args(self) -> None:
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        assert CrewAgentExecutor._resolve_file_references({}) == {}


class TestModuleSingleton:
    def test_singleton_is_shared(self) -> None:
        from crewai.tools.tool_file_reference import (
            tool_file_store as store1,
        )
        from crewai.tools.tool_file_reference import (
            tool_file_store as store2,
        )

        assert store1 is store2
