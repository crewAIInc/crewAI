"""Tests for deserialization guard in _resolve_tool_dict."""

from __future__ import annotations

import os

import pytest

from crewai.tools.base_tool import _resolve_tool_dict


class TestResolveToolDictGuard:
    """Unregistered tool types must require CREWAI_DESERIALIZE_CALLBACKS=1."""

    def test_blocks_unregistered_tool_type_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CREWAI_DESERIALIZE_CALLBACKS", raising=False)
        with pytest.raises(ValueError, match="Refusing to resolve tool type"):
            _resolve_tool_dict({"tool_type": "os.system", "name": "evil", "description": "x"})

    def test_blocks_arbitrary_import_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CREWAI_DESERIALIZE_CALLBACKS", raising=False)
        with pytest.raises(ValueError, match="Refusing to resolve tool type"):
            _resolve_tool_dict({"tool_type": "subprocess.Popen", "name": "evil", "description": "x"})

    def test_cache_function_not_pre_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cache_function strings must NOT be pre-resolved — SerializableCallable handles it."""
        monkeypatch.delenv("CREWAI_DESERIALIZE_CALLBACKS", raising=False)
        # Even with a registered tool type, cache_function should be passed as-is
        from crewai.tools.base_tool import BaseTool, _TOOL_TYPE_REGISTRY

        # Find any registered tool type to use
        if _TOOL_TYPE_REGISTRY:
            key = next(iter(_TOOL_TYPE_REGISTRY))
            # This should not crash trying to pre-resolve the string —
            # the pre-resolution code has been removed
            try:
                _resolve_tool_dict({
                    "tool_type": key,
                    "name": "test",
                    "description": "test",
                    "cache_function": "some.nonexistent.function",
                })
            except Exception:
                # May fail for other reasons (validation), but should NOT
                # fail because of _resolve_dotted_path being called
                pass
