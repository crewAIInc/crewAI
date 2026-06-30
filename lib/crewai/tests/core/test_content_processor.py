"""Unit tests for the content processor provider module."""

from __future__ import annotations

import contextvars
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.core.providers.content_processor import (
    ContentProcessorProvider,
    NoOpContentProcessor,
    _content_processor,
    _default_processor,
    get_processor,
    process_content,
    set_processor,
)


@pytest.fixture(autouse=True)
def _reset_processor_context() -> None:
    """Reset the context variable to None before and after each test."""
    token = _content_processor.set(None)
    yield
    _content_processor.reset(token)


class TestNoOpContentProcessor:
    """Tests for the NoOpContentProcessor default implementation."""

    def test_returns_content_unchanged(self) -> None:
        processor = NoOpContentProcessor()
        result = processor.process("hello world")
        assert result == "hello world"

    def test_returns_empty_string(self) -> None:
        processor = NoOpContentProcessor()
        assert processor.process("") == ""

    def test_returns_multiline_content(self) -> None:
        processor = NoOpContentProcessor()
        content = "line 1\nline 2\nline 3"
        assert processor.process(content) == content

    def test_returns_content_with_special_characters(self) -> None:
        processor = NoOpContentProcessor()
        content = "<script>alert('xss')</script>"
        assert processor.process(content) == content

    def test_ignores_context_parameter(self) -> None:
        processor = NoOpContentProcessor()
        result = processor.process("hello", context={"key": "value"})
        assert result == "hello"

    def test_context_none_returns_same_content(self) -> None:
        processor = NoOpContentProcessor()
        assert processor.process("data", context=None) == "data"


class TestContentProcessorProviderProtocol:
    """Tests for the ContentProcessorProvider Protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        # isinstance() raises TypeError if the Protocol is not @runtime_checkable,
        # so this single assertion properly verifies runtime-checkability.
        assert isinstance(NoOpContentProcessor(), ContentProcessorProvider)

    def test_noop_processor_satisfies_protocol(self) -> None:
        processor = NoOpContentProcessor()
        assert isinstance(processor, ContentProcessorProvider)

    def test_custom_class_with_process_satisfies_protocol(self) -> None:
        class CustomProcessor:
            def process(
                self, content: str, context: dict[str, Any] | None = None
            ) -> str:
                return content.upper()

        assert isinstance(CustomProcessor(), ContentProcessorProvider)

    def test_class_without_process_does_not_satisfy_protocol(self) -> None:
        class NotAProcessor:
            def transform(self, content: str) -> str:
                return content

        assert not isinstance(NotAProcessor(), ContentProcessorProvider)


class TestGetProcessor:
    """Tests for the get_processor function."""

    def test_returns_noop_by_default(self) -> None:
        processor = get_processor()
        assert isinstance(processor, NoOpContentProcessor)

    def test_returns_same_default_instance(self) -> None:
        """Multiple calls return the same default NoOpContentProcessor instance."""
        assert get_processor() is _default_processor
        assert get_processor() is _default_processor

    def test_returns_set_processor(self) -> None:
        custom = MagicMock(spec=ContentProcessorProvider)
        set_processor(custom)
        assert get_processor() is custom

    def test_returns_noop_after_reset(self) -> None:
        custom = MagicMock(spec=ContentProcessorProvider)
        set_processor(custom)
        _content_processor.set(None)
        assert isinstance(get_processor(), NoOpContentProcessor)


class TestSetProcessor:
    """Tests for the set_processor function."""

    def test_set_and_get_processor(self) -> None:
        custom = MagicMock(spec=ContentProcessorProvider)
        set_processor(custom)
        assert get_processor() is custom

    def test_set_processor_replaces_previous(self) -> None:
        first = MagicMock(spec=ContentProcessorProvider)
        second = MagicMock(spec=ContentProcessorProvider)
        set_processor(first)
        set_processor(second)
        assert get_processor() is second

    def test_set_processor_does_not_affect_default_instance(self) -> None:
        custom = MagicMock(spec=ContentProcessorProvider)
        set_processor(custom)
        # The default processor object is unchanged
        assert isinstance(_default_processor, NoOpContentProcessor)


class TestProcessContent:
    """Tests for the process_content convenience function."""

    def test_uses_noop_by_default(self) -> None:
        result = process_content("hello")
        assert result == "hello"

    def test_passes_context_to_processor(self) -> None:
        custom = MagicMock(spec=ContentProcessorProvider)
        custom.process.return_value = "processed"
        set_processor(custom)

        context: dict[str, Any] = {"key": "value"}
        process_content("input", context)

        custom.process.assert_called_once_with("input", context)

    def test_passes_none_context_by_default(self) -> None:
        custom = MagicMock(spec=ContentProcessorProvider)
        custom.process.return_value = "output"
        set_processor(custom)

        process_content("input")

        custom.process.assert_called_once_with("input", None)

    def test_custom_processor_transforms_content(self) -> None:
        class UpperProcessor:
            def process(
                self, content: str, context: dict[str, Any] | None = None
            ) -> str:
                return content.upper()

        set_processor(UpperProcessor())
        assert process_content("hello") == "HELLO"

    def test_return_value_propagated(self) -> None:
        custom = MagicMock(spec=ContentProcessorProvider)
        custom.process.return_value = "transformed"
        set_processor(custom)

        assert process_content("raw") == "transformed"


class TestContextVarIsolation:
    """Tests for ContextVar-based isolation of processor state."""

    def test_context_copy_isolation(self) -> None:
        """Processors set in a copied context do not leak to the parent."""
        results: dict[str, str] = {}

        custom = MagicMock(spec=ContentProcessorProvider)
        custom.process.return_value = "isolated"

        def isolated_task() -> None:
            set_processor(custom)
            results["child"] = process_content("input")

        ctx = contextvars.copy_context()
        ctx.run(isolated_task)

        # Parent context is unaffected — still uses the default processor
        results["parent"] = process_content("input")

        assert results["child"] == "isolated"
        assert results["parent"] == "input"

    def test_thread_isolation(self) -> None:
        """Processors set in one thread do not leak to another."""
        results: dict[str, str] = {}
        barrier = threading.Barrier(2)

        def thread_a() -> None:
            custom_a = MagicMock(spec=ContentProcessorProvider)
            custom_a.process.return_value = "thread_a"
            set_processor(custom_a)
            barrier.wait()
            results["a"] = process_content("input")

        def thread_b() -> None:
            custom_b = MagicMock(spec=ContentProcessorProvider)
            custom_b.process.return_value = "thread_b"
            set_processor(custom_b)
            barrier.wait()
            results["b"] = process_content("input")

        t_a = threading.Thread(target=thread_a)
        t_b = threading.Thread(target=thread_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)
        assert not t_a.is_alive()
        assert not t_b.is_alive()

        assert results["a"] == "thread_a"
        assert results["b"] == "thread_b"

    def test_processor_not_shared_across_contexts(self) -> None:
        """Main-thread processor does not leak into a spawned thread."""
        custom_main = MagicMock(spec=ContentProcessorProvider)
        custom_main.process.return_value = "main"
        set_processor(custom_main)

        child_result: list[str] = []

        def child_thread() -> None:
            # No processor set in child; should fall back to default
            child_result.append(process_content("input"))

        t = threading.Thread(target=child_thread)
        t.start()
        t.join()

        # Child thread sees the default NoOpContentProcessor, not custom_main
        assert child_result[0] == "input"
