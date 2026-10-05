"""Unit tests for the human-feedback prompt (issue #6072).

The feedback panel must show the result under review regardless of the
``verbose`` setting, so the prompt never references output that was not
displayed.
"""

from __future__ import annotations

import inspect


def test_prompt_functions_accept_result():
    from crewai.core.providers.human_input import SyncHumanInputProvider

    assert "result" in inspect.signature(SyncHumanInputProvider._prompt_input).parameters
    assert (
        "result"
        in inspect.signature(SyncHumanInputProvider._prompt_input_async).parameters
    )


def test_get_output_string_handles_str_and_model():
    from crewai.core.providers.human_input import HumanInputProvider

    class FakeModel:
        def model_dump_json(self):
            return '{"text": "hi"}'

    class FakeFinish:
        def __init__(self, output):
            self.output = output

    assert HumanInputProvider._get_output_string(FakeFinish("hello")) == "hello"
    dumped = HumanInputProvider._get_output_string(FakeFinish(FakeModel()))
    assert '"hi"' in dumped


def test_prompt_embeds_result_instead_of_referencing_above():
    from crewai.core.providers.human_input import SyncHumanInputProvider

    src = inspect.getsource(SyncHumanInputProvider._prompt_input)
    assert "Final Output:" in src
    assert "Final Result above" not in src
