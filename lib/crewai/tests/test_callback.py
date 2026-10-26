"""Tests for crewai.types.callback — SerializableCallable round-tripping."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any
import pytest
from pydantic import BaseModel, ValidationError

from crewai.types.callback import (
    SerializableCallable,
    _resolve_dotted_path,
    callable_to_string,
    string_to_callable,
)




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

    @classmethod
    def class_method(cls) -> str:
        return "class_method"

    class_body_lambda = lambda: None  # noqa: E731


class _Model(BaseModel):
    cb: SerializableCallable | None = None




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

    def test_lambda_returns_none(self) -> None:
        fn = lambda: None  # noqa: E731
        assert callable_to_string(fn) is None

    def test_closure_returns_none(self) -> None:
        def outer() -> Callable[[], None]:
            def inner() -> None:
                return None

            return inner

        assert callable_to_string(outer()) is None

    def test_lambda_in_class_body_returns_none(self) -> None:
        assert callable_to_string(_HasMethod.class_body_lambda) is None

    def test_bound_instance_method_returns_none(self) -> None:
        assert callable_to_string(_HasMethod().method) is None

    def test_classmethod_serializes_to_dotted_path(self) -> None:
        assert (
            callable_to_string(_HasMethod.class_method)
            == f"{__name__}._HasMethod.class_method"
        )

    def test_partial_returns_none(self) -> None:
        assert callable_to_string(functools.partial(print, "hi")) is None

    def test_callable_instance_returns_none(self) -> None:
        assert callable_to_string(_CallableInstance()) is None

    def test_missing_qualname_returns_none(self) -> None:
        obj = type("NoQual", (), {"__module__": "test"})()
        obj.__qualname__ = None  # type: ignore[assignment]
        assert callable_to_string(obj) is None

    def test_missing_module_returns_none(self) -> None:
        ns: dict[str, Any] = {"__qualname__": "x", "__module__": None}
        obj = type("NoMod", (), ns)()
        assert callable_to_string(obj) is None




class TestStringToCallable:
    def test_callable_passthrough(self) -> None:
        assert string_to_callable(print) is print

    @pytest.mark.parametrize(
        "fn",
        [print, dict, module_level_function, _HasMethod.class_method],
        ids=["builtin", "class", "module_function", "classmethod"],
    )
    def test_roundtrippable_callable_no_warning(
        self, fn: Callable[..., Any], recwarn: pytest.WarningsChecker
    ) -> None:
        string_to_callable(fn)
        our_warnings = [
            w for w in recwarn if "cannot be saved in a checkpoint" in str(w.message)
        ]
        assert our_warnings == []

    def test_lambda_warns_it_is_dropped(self) -> None:
        with pytest.warns(
            UserWarning,
            match="A function callback cannot be saved in a checkpoint; "
            "it is dropped and will not run after a restore",
        ):
            string_to_callable(lambda: None)

    def test_closure_warns_it_is_dropped(self) -> None:
        x = 1

        def closure() -> int:
            return x

        with pytest.warns(UserWarning, match="dropped and will not run after a restore"):
            string_to_callable(closure)

    def test_bound_method_warns_it_is_dropped(self) -> None:
        with pytest.warns(
            UserWarning,
            match="A method callback cannot be saved in a checkpoint; "
            "it is dropped and will not run after a restore",
        ):
            string_to_callable(_HasMethod().method)

    def test_partial_warns_it_is_dropped(self) -> None:
        with pytest.warns(
            UserWarning,
            match="A partial callback cannot be saved in a checkpoint; "
            "it is dropped and will not run after a restore",
        ):
            string_to_callable(functools.partial(print))

    def test_callable_instance_warns_it_is_dropped(self) -> None:
        with pytest.warns(
            UserWarning,
            match="A _CallableInstance callback cannot be saved in a checkpoint",
        ):
            string_to_callable(_CallableInstance())

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

    def test_json_round_trip_classmethod(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CREWAI_DESERIALIZE_CALLBACKS", "1")
        m = _Model(cb=_HasMethod.class_method)
        restored = _Model.model_validate_json(m.model_dump_json())
        assert restored.cb is not None
        assert restored.cb() == "class_method"

    @pytest.mark.parametrize(
        "make_cb",
        [
            lambda: _HasMethod().method,
            lambda: functools.partial(print, "hi"),
            lambda: _CallableInstance(),
        ],
        ids=["bound_method", "partial", "callable_instance"],
    )
    def test_json_dumps_unrestorable_callable_as_null(
        self, make_cb: Callable[[], Callable[..., Any]]
    ) -> None:
        with pytest.warns(UserWarning, match="dropped and will not run"):
            m = _Model(cb=make_cb())
        assert m.model_dump(mode="json") == {"cb": None}
        assert _Model.model_validate_json(m.model_dump_json()).cb is None

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