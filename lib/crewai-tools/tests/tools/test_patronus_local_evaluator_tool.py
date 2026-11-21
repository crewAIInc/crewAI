from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.patronus_eval_tool import patronus_local_evaluator_tool


@pytest.fixture
def patronus_client() -> MagicMock:
    client = MagicMock()
    client.evaluate.return_value = MagicMock(pass_=True, explanation="ok")
    return client


def test_constructor_accepts_configured_evaluator(
    monkeypatch: pytest.MonkeyPatch, patronus_client: MagicMock
) -> None:
    """The documented constructor arguments must produce a usable tool.

    ``evaluator`` and ``evaluated_model_gold_answer`` are required model fields,
    but the constructor kept the values to itself instead of forwarding them to
    ``super().__init__``, so every construction raised a ``ValidationError``.
    """
    monkeypatch.setattr(patronus_local_evaluator_tool, "PYPATRONUS_AVAILABLE", True)

    tool = patronus_local_evaluator_tool.PatronusLocalEvaluatorTool(
        patronus_client=patronus_client,
        evaluator="random_evaluator",
        evaluated_model_gold_answer="example label",
    )

    assert tool.evaluator == "random_evaluator"
    assert tool.evaluated_model_gold_answer == "example label"


def test_run_uses_configured_evaluator(
    monkeypatch: pytest.MonkeyPatch, patronus_client: MagicMock
) -> None:
    """The configured evaluator and gold answer are sent to the client."""
    monkeypatch.setattr(patronus_local_evaluator_tool, "PYPATRONUS_AVAILABLE", True)
    tool = patronus_local_evaluator_tool.PatronusLocalEvaluatorTool(
        patronus_client=patronus_client,
        evaluator="random_evaluator",
        evaluated_model_gold_answer="example label",
    )

    result = tool.run(
        evaluated_model_input="question",
        evaluated_model_output="answer",
        evaluated_model_retrieved_context="context",
        evaluated_model_gold_answer="per-call label",
        evaluator="per-call evaluator",
    )

    assert "Evaluation result: True" in result
    call_kwargs = patronus_client.evaluate.call_args.kwargs
    assert call_kwargs["evaluator"] == "random_evaluator"
    assert call_kwargs["evaluated_model_gold_answer"] == "example label"
