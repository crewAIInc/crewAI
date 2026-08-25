"""A model-call deny must reach the caller as a deny.

Two layers used to erase it. The LLM layer caught ``HookAborted`` and returned
``False``, which every provider turned into ``ValueError("LLM call blocked...")``
— losing the reason, the source, and any way to tell a policy decision from a
provider outage. Downstream, every internal model call is wrapped in
``except Exception`` so a provider hiccup degrades instead of failing the run,
and those handlers then absorbed the flattened deny, often retrying the very
call that was just denied.

These use a real ``LLM`` with a real ``pre_model_call`` hook: the deny fires
inside ``dispatch`` before the provider is reached, so nothing here needs a
network or a cassette.
"""

from __future__ import annotations

from typing import Any

from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
    clear_all,
    on,
)
from crewai.hooks.llm_hooks import register_before_llm_call_hook
from crewai.llm import LLM
from crewai.memory.analyze import (
    analyze_for_consolidation,
    analyze_for_save,
    analyze_query,
    extract_memories_from_content,
)
from crewai.memory.types import MemoryRecord
from crewai.utilities.converter import Converter
import pytest
from pydantic import BaseModel


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_all()
    yield
    clear_all()


@pytest.fixture
def denying_llm() -> LLM:
    denied: list[Any] = []

    @on(InterceptionPoint.PRE_MODEL_CALL)
    def deny(ctx):
        denied.append(ctx)
        raise HookAborted(reason="no model calls allowed", source="policy")

    llm = LLM(model="gpt-4o-mini")
    llm.denied_calls = denied  # type: ignore[attr-defined]
    return llm


class _Person(BaseModel):
    name: str


def test_a_raised_deny_keeps_its_reason_out_of_the_llm_layer(denying_llm):
    with pytest.raises(HookAborted) as exc:
        denying_llm.call([{"role": "user", "content": "hi"}])

    assert exc.value.reason == "no model calls allowed"
    assert exc.value.source == "policy"


def test_the_boolean_convention_still_blocks_with_the_documented_error():
    register_before_llm_call_hook(lambda _ctx: False)

    with pytest.raises(ValueError, match="LLM call blocked by before_llm_call hook"):
        LLM(model="gpt-4o-mini").call([{"role": "user", "content": "hi"}])


class TestMemoryAnalysis:
    """The four helpers behind memory save and recall."""

    def test_a_deny_reaches_the_caller_instead_of_a_safe_default(self, denying_llm):
        cases = {
            "extract": lambda: extract_memories_from_content(
                "some content", denying_llm
            ),
            "query": lambda: analyze_query("a query", ["/"], None, denying_llm),
            "save": lambda: analyze_for_save("some content", ["/"], [], denying_llm),
            "consolidate": lambda: analyze_for_consolidation(
                "new content",
                [MemoryRecord(id="1", content="old content", scope="/")],
                denying_llm,
            ),
        }
        for name, call in cases.items():
            with pytest.raises(HookAborted):
                call()
            assert denying_llm.denied_calls, f"{name} never reached the model"
            denying_llm.denied_calls.clear()


def test_a_denied_conversion_is_not_retried(denying_llm):
    converter = Converter(
        text="Name: Ada",
        llm=denying_llm,
        model=_Person,
        instructions="Extract the person",
        max_attempts=3,
    )

    with pytest.raises(HookAborted):
        converter.to_pydantic()

    assert len(denying_llm.denied_calls) == 1
