from collections.abc import Callable
from functools import partial, wraps
from typing import Annotated, Any

import pytest
from crewai import Task, TaskOutput
from pydantic import ValidationError
from typing_extensions import NotRequired, Required


def make_guardrail(
    annotation: str, postponed: bool, kind: str = "function"
) -> Callable[..., Any]:
    prefix = "from __future__ import annotations\n" if postponed else ""
    suffix = f" -> {annotation}" if annotation else ""
    if kind == "function":
        code = prefix + f"def guardrail(output){suffix}:\n    return True, output\n"
    else:
        code = (
            prefix
            + f"class Guardrail:\n    def __call__(self, output){suffix}:\n        return True, output\nguardrail = Guardrail()\n"
        )
    namespace = {
        "Any": Any,
        "TaskOutput": TaskOutput,
        "Annotated": Annotated,
        "Required": Required,
        "NotRequired": NotRequired,
    }
    exec(compile(code, "<guardrail_annotations>", "exec", dont_inherit=True), namespace)
    return namespace["guardrail"]


@pytest.mark.parametrize("postponed", [False, True], ids=["eager", "postponed"])
@pytest.mark.parametrize("kind", ["function", "callable_object"])
@pytest.mark.parametrize(
    "annotation",
    [
        "",
        "tuple[bool, Any]",
        "tuple[bool, str]",
        "tuple[bool, TaskOutput]",
        "tuple[bool, str | TaskOutput]",
    ],
)
def test_valid_guardrail_annotation(
    annotation: str, postponed: bool, kind: str
) -> None:
    guardrail = make_guardrail(annotation, postponed, kind)
    task = Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=guardrail,
    )
    assert task.guardrail is guardrail
    assert task._guardrail is guardrail


@pytest.mark.parametrize("postponed", [False, True], ids=["eager", "postponed"])
@pytest.mark.parametrize(
    "annotation",
    [
        "bool",
        "tuple[bool, int]",
        "tuple[int, Any]",
        "tuple[bool]",
        "tuple[bool, Any, str]",
        "Annotated[tuple[bool, Any], 'metadata']",
        "tuple[bool, Annotated[str, 'metadata']]",
        "tuple[bool, Required[str]]",
        "tuple[bool, NotRequired[str]]",
    ],
)
def test_invalid_annotation_still_rejected(annotation: str, postponed: bool) -> None:
    guardrail = make_guardrail(annotation, postponed)
    with pytest.raises(ValidationError, match="If return type is annotated"):
        Task(
            description="Validate a short answer",
            expected_output="A string",
            guardrail=guardrail,
        )


def test_invalid_signature_still_rejected() -> None:
    def guardrail(first: Any, second: Any) -> tuple[bool, Any]:
        return True, first

    with pytest.raises(ValidationError, match="exactly one parameter"):
        Task(
            description="Validate a short answer",
            expected_output="A string",
            guardrail=guardrail,
        )


def test_unresolved_annotation_still_rejected() -> None:
    guardrail = make_guardrail("MissingReturnType", True)
    with pytest.raises(ValidationError, match="If return type is annotated"):
        Task(
            description="Validate a short answer",
            expected_output="A string",
            guardrail=guardrail,
        )


def test_eager_return_with_unresolved_input_preserved() -> None:
    def guardrail(output: "UnavailableInputType") -> tuple[bool, Any]:  # noqa: F821
        return True, output

    task = Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=guardrail,
    )
    assert task.guardrail is guardrail


def test_malformed_string_annotation_still_rejected() -> None:
    guardrail = make_guardrail("'tuple[bool,'", False)
    with pytest.raises(ValidationError, match="If return type is annotated"):
        Task(
            description="Validate a short answer",
            expected_output="A string",
            guardrail=guardrail,
        )


@pytest.mark.parametrize("annotation", ["", "tuple[bool, Any]"])
@pytest.mark.parametrize("input_hint", ["'typing.DoesNotExist'", "'1 / 0'"])
def test_irrelevant_eager_input_annotations_are_not_evaluated(
    annotation: str, input_hint: str
) -> None:
    suffix = f" -> {annotation}" if annotation else ""
    namespace = {"Any": Any}
    code = f"import typing\ndef guardrail(output: {input_hint}){suffix}:\n    return True, output\n"
    exec(compile(code, "<eager_input>", "exec", dont_inherit=True), namespace)
    Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=namespace["guardrail"],
    )


def test_postponed_return_with_unavailable_input_type() -> None:
    namespace = {"Any": Any}
    code = "from __future__ import annotations\ndef guardrail(output: TypeCheckingOnly) -> tuple[bool, Any]:\n    return True, output\n"
    exec(compile(code, "<postponed_input>", "exec", dont_inherit=True), namespace)
    Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=namespace["guardrail"],
    )


@pytest.mark.parametrize("postponed", [False, True])
def test_explicitly_quoted_valid_return(postponed: bool) -> None:
    guardrail = make_guardrail("'tuple[bool, Any]'", postponed)
    Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=guardrail,
    )


def test_original_annotations_are_not_mutated() -> None:
    guardrail = make_guardrail("tuple[bool, Any]", True)
    annotations = guardrail.__annotations__
    original = dict(annotations)
    Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=guardrail,
    )
    assert guardrail.__annotations__ is annotations
    assert guardrail.__annotations__ == original


@pytest.mark.parametrize("annotated_return", [False, True])
def test_input_annotation_expressions_are_not_evaluated(annotated_return: bool) -> None:
    evaluations = []

    def input_type() -> Any:
        evaluations.append(True)
        return Any

    namespace = {"Any": Any, "input_type": input_type}
    suffix = " -> tuple[bool, Any]" if annotated_return else ""
    source = (
        "from __future__ import annotations\n"
        + f"def guardrail(output: input_type()){suffix}:\n    return True, output\n"
    )
    exec(compile(source, "<input_expressions>", "exec", dont_inherit=True), namespace)
    Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=namespace["guardrail"],
    )
    assert evaluations == []


def test_decorated_callable_method_resolves_defining_namespace() -> None:
    defining = {"ReturnAlias": tuple[bool, str]}
    original_source = "from __future__ import annotations\ndef original(self, output) -> ReturnAlias:\n    return True, output\n"
    exec(
        compile(original_source, "<defining_namespace>", "exec", dont_inherit=True),
        defining,
    )
    decoration = {"wraps": wraps, "original": defining["original"]}
    wrapper_source = "@wraps(original)\ndef wrapper(*args, **kwargs):\n    return original(*args, **kwargs)\n"
    exec(
        compile(wrapper_source, "<decorator_namespace>", "exec", dont_inherit=True),
        decoration,
    )
    guardrail_class = type("Guardrail", (), {"__call__": decoration["wrapper"]})
    guardrail = guardrail_class()
    task = Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=guardrail,
    )
    assert task.guardrail is guardrail


def test_partial_callable_instance_resolves_defining_namespace() -> None:
    defining = {"ReturnAlias": tuple[bool, str]}
    source = "from __future__ import annotations\nclass Guardrail:\n    def __call__(self, prefix, output) -> ReturnAlias:\n        return True, prefix + output\n"
    exec(compile(source, "<partial_namespace>", "exec", dont_inherit=True), defining)
    guardrail = partial(defining["Guardrail"](), "prefix")
    task = Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=guardrail,
    )
    assert task.guardrail is guardrail


@pytest.mark.parametrize("kind", ["bound_method", "partial", "wrapped"])
def test_postponed_callable_wrappers(kind: str) -> None:
    namespace = {"Any": Any, "partial": partial, "wraps": wraps}
    bodies = {
        "bound_method": "class Guardrail:\n    def validate(self, output) -> tuple[bool, Any]:\n        return True, output\nguardrail = Guardrail().validate\n",
        "partial": "def original(prefix, output) -> tuple[bool, Any]:\n    return True, output\nguardrail = partial(original, 'prefix')\n",
        "wrapped": "def original(output) -> tuple[bool, Any]:\n    return True, output\n@wraps(original)\ndef guardrail(*args, **kwargs):\n    return original(*args, **kwargs)\n",
    }
    source = "from __future__ import annotations\n" + bodies[kind]
    exec(compile(source, "<callable_wrappers>", "exec", dont_inherit=True), namespace)
    Task(
        description="Validate a short answer",
        expected_output="A string",
        guardrail=namespace["guardrail"],
    )
