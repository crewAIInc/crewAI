"""LiteLLM fallback must load on releases without newer optional types."""

import builtins

import pytest

import crewai.llm as llm_module

_LITELLM_GLOBALS = (
    "litellm",
    "Choices",
    "LiteLLMDelta",
    "Message",
    "ModelResponseBase",
    "ModelResponseStream",
    "LiteLLMStreamingChoices",
    "get_supported_openai_params",
    "ChatCompletionDeltaCustomToolCall",
    "ChatCompletionDeltaToolCall",
    "Function",
    "ModelResponse",
    "supports_response_schema",
)


def test_litellm_loads_without_custom_tool_call_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older litellm lacks ChatCompletionDeltaCustomToolCall; it is only a type hint."""
    pytest.importorskip("litellm")

    original_import = builtins.__import__

    def import_without_custom_tool_call(
        name: str, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "litellm.types.utils" and "ChatCompletionDeltaCustomToolCall" in (
            fromlist or ()
        ):
            raise ImportError(
                "cannot import name 'ChatCompletionDeltaCustomToolCall' "
                "from 'litellm.types.utils'"
            )
        return original_import(name, globals, locals, fromlist, level)

    for attr in _LITELLM_GLOBALS:
        monkeypatch.setattr(llm_module, attr, None)
    monkeypatch.setattr(llm_module, "_litellm_loaded", False)
    monkeypatch.setattr(llm_module, "LITELLM_AVAILABLE", False)
    monkeypatch.setattr(builtins, "__import__", import_without_custom_tool_call)

    assert llm_module._ensure_litellm() is True
    assert llm_module.LITELLM_AVAILABLE is True
    assert llm_module.litellm is not None
    assert llm_module.ChatCompletionDeltaToolCall is not None
    assert llm_module.ChatCompletionDeltaCustomToolCall is None
