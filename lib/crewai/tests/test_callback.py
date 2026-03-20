"""Tests for crewai.types.callback — SerializableCallable round-tripping."""

from __future__ import annotations

import functools
import os
from typing import Any
import pytest
from pydantic import BaseModel, ValidationError

from crewai.types.callback import (
    SerializableCallable,
    _is_non_roundtrippable,
    _resolve_dotted_path,
    callable_to_string,
    string_to_callable,
)


# ── Helpers ──────────────────────────────────────────────────────────


def module_level_function() -> str:
    """Plain module-level function that should round-trip."""
    return "hello"


class _CallableInstance:
    """Callable class instance — non-roundtrippable."""

    def __call__(self) -> str:
        return "instance"


class _HasMethod:
    def method(self) -> str:
        return "method"


class _Model(BaseModel):
    cb: SerializableCallable | None = None


# ── _is_non_roundtrippable ───────────────────────────────────────────


class TestIsNonRoundtrippable:
    def test_builtin_is_roundtrippable(self) -> None:
        assert _is_non_roundtrippable(print) is False
        assert _is_non_roundtrippable(len) is False

    def test_class_is_roundtrippable(self) -> None:
        assert _is_non_roundtrippable(dict) is False
        assert _is_non_roundtrippable(_CallableInstance) is False

    def test_module_level_function_is_roundtrippable(self) -> None:
        assert _is_non_roundtrippable(module_level_function) is False

    def test_lambda_is_non_roundtrippable(self) -> None:
        assert _is_non_roundtrippable(lambda: None) is True

    def test_closure_is_non_roundtrippable(self) -> None:
        x = 1

        def closure() -> int:
            return x

        assert _is_non_roundtrippable(closure) is True

    def test_bound_method_is_non_roundtrippable(self) -> None:
        assert _is_non_roundtrippable(_HasMethod().method) is True

    def test_partial_is_non_roundtrippable(self) -> None:
        assert _is_non_roundtrippable(functools.partial(print, "hi")) is True

    def test_callable_instance_is_non_roundtrippable(self) -> None:
        assert _is_non_roundtrippable(_CallableInstance()) is True


# ── callable_to_string ───────────────────────────────────────────────


class TestCallableToString:
    def test_module_level_function(self) -> None:
        result = callable_to_string(module_level_function)
        assert result == f"{__name__}.module_level_function"

    def test_class(self) -> None:
        result = callable_to_string(dict)
        assert result == "builtins.dict"

    def test_builtin(self) -> None:
        result = callable_to_string(print)
        assert result == "builtins.print"

    def test_lambda_produces_locals_path(self) -> None:
        fn = lambda: None  # noqa: E731
        result = callable_to_string(fn)
        assert "<lambda>" in result

    def test_missing_qualname_raises(self) -> None:
        obj = type("NoQual", (), {"__module__": "test"})()
        obj.__qualname__ = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="missing __module__ or __qualname__"):
            callable_to_string(obj)

    def test_missing_module_raises(self) -> None:
        # Create an object where getattr(obj, "__module__", None) returns None
        ns: dict[str, Any] = {"__qualname__": "x", "__module__": None}
        obj = type("NoMod", (), ns)()
        with pytest.raises(ValueError, match="missing __module__"):
            callable_to_string(obj)


# ── string_to_callable ───────────────────────────────────────────────


class TestStringToCallable:
    def test_callable_passthrough(self) -> None:
        assert string_to_callable(print) is print

    def test_roundtrippable_callable_no_warning(self, recwarn: pytest.WarningsChecker) -> None:
        string_to_callable(module_level_function)
        our_warnings = [
            w for w in recwarn if "cannot be serialized" in str(w.message)
        ]
        assert our_warnings == []

    def test_non_roundtrippable_warns(self) -> None:
        with pytest.warns(UserWarning, match="cannot be serialized"):
            string_to_callable(functools.partial(print))

    def test_non_callable_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected a callable"):
            string_to_callable(42)

    def test_string_without_dot_raises(self) -> None:
        with pytest.raises(ValueError, match="expected 'module.name' format"):
            string_to_callable("nodots")

    def test_string_refused_without_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CREWAI_DESERIALIZE_CALLBACKS", raising=False)
        with pytest.raises(ValueError, match="Refusing to resolve"):
            string_to_callable("builtins.print")

    def test_string_resolves_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CREWAI_DESERIALIZE_CALLBACKS", "1")
        result = string_to_callable("builtins.print")
        assert result is print

    def test_string_resolves_multi_level_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CREWAI_DESERIALIZE_CALLBACKS", "1")
        result = string_to_callable("os.path.join")
        assert result is os.path.join

    def test_unresolvable_path_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CREWAI_DESERIALIZE_CALLBACKS", "1")
        with pytest.raises(ValueError, match="Cannot resolve"):
            string_to_callable("nonexistent.module.func")


# ── _resolve_dotted_path ─────────────────────────────────────────────


class TestResolveDottedPath:
    def test_builtin(self) -> None:
        assert _resolve_dotted_path("builtins.print") is print

    def test_nested_module_attribute(self) -> None:
        assert _resolve_dotted_path("os.path.join") is os.path.join

    def test_class_on_module(self) -> None:
        from collections import OrderedDict

        assert _resolve_dotted_path("collections.OrderedDict") is OrderedDict

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot resolve"):
            _resolve_dotted_path("no.such.module.func")

    def test_non_callable_attribute_skipped(self) -> None:
        # os.sep is a string, not callable — should not resolve
        with pytest.raises(ValueError, match="Cannot resolve"):
            _resolve_dotted_path("os.sep")


# ── Pydantic integration round-trip ──────────────────────────────────


class TestSerializableCallableRoundTrip:
    def test_json_serialize_module_function(self) -> None:
        m = _Model(cb=module_level_function)
        data = m.model_dump(mode="json")
        assert data["cb"] == f"{__name__}.module_level_function"

    def test_json_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CREWAI_DESERIALIZE_CALLBACKS", "1")
        m = _Model(cb=print)
        json_str = m.model_dump_json()
        restored = _Model.model_validate_json(json_str)
        assert restored.cb is print

    def test_json_round_trip_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CREWAI_DESERIALIZE_CALLBACKS", "1")
        m = _Model(cb=dict)
        json_str = m.model_dump_json()
        restored = _Model.model_validate_json(json_str)
        assert restored.cb is dict

    def test_python_mode_preserves_callable(self) -> None:
        m = _Model(cb=module_level_function)
        data = m.model_dump(mode="python")
        assert data["cb"] is module_level_function

    def test_none_field(self) -> None:
        m = _Model(cb=None)
        assert m.cb is None
        data = m.model_dump(mode="json")
        assert data["cb"] is None

    def test_validation_error_for_int(self) -> None:
        with pytest.raises(ValidationError):
            _Model(cb=42)  # type: ignore[arg-type]

    def test_deserialization_refused_without_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CREWAI_DESERIALIZE_CALLBACKS", raising=False)
        with pytest.raises(ValidationError, match="Refusing to resolve"):
            _Model.model_validate({"cb": "builtins.print"})

    def test_json_schema_is_string(self) -> None:
        schema = _Model.model_json_schema()
        cb_schema = schema["properties"]["cb"]
        # anyOf for Optional: one string, one null
        types = {item.get("type") for item in cb_schema.get("anyOf", [cb_schema])}
        assert "string" in types