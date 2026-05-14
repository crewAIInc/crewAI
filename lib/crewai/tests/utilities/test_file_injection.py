"""Tests for deciding which files are injected into LLM messages."""

from __future__ import annotations

from typing import Any

from crewai.llms.base_llm import BaseLLM
from crewai.utilities.file_injection import get_auto_injected_files
from crewai_files import FileBytes, ImageFile, TextFile


class DummyLLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o", multimodal: bool = True) -> None:
        super().__init__(model=model)
        self._multimodal = multimodal

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[Any] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
    ) -> str:
        return "ok"

    def supports_multimodal(self) -> bool:
        return self._multimodal


def test_text_files_are_not_injected_for_non_multimodal_llm() -> None:
    files = {"readme": TextFile(source=FileBytes(data=b"hello", filename="readme.md"))}
    llm = DummyLLM(multimodal=False)

    assert get_auto_injected_files(files, llm) == {}


def test_unsupported_file_values_are_skipped() -> None:
    files = {"bad": object()}
    llm = DummyLLM(model="openai/gpt-4o", multimodal=True)

    assert get_auto_injected_files(files, llm) == {}


def test_only_supported_files_are_injected_for_multimodal_llm() -> None:
    image = ImageFile(source=FileBytes(data=b"\x89PNG\r\n\x1a\n", filename="chart.png"))
    text = TextFile(source=FileBytes(data=b"hello", filename="readme.md"))
    files = {"chart": image, "readme": text}
    llm = DummyLLM(model="openai/gpt-4o", multimodal=True)

    assert get_auto_injected_files(files, llm) == {"chart": image}
