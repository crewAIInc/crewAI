from unittest.mock import MagicMock, patch

from crewai.utilities.converter import ConverterError
from crewai.utilities.evaluators.task_evaluator import (
    TaskEvaluator,
    TrainingTaskEvaluation,
)


@patch("crewai.utilities.evaluators.task_evaluator.TrainingConverter")
def test_evaluate_training_data(converter_mock):
    training_data = {
        "agent_id": {
            "data1": {
                "initial_output": "Initial output 1",
                "human_feedback": "Human feedback 1",
                "improved_output": "Improved output 1",
            },
            "data2": {
                "initial_output": "Initial output 2",
                "human_feedback": "Human feedback 2",
                "improved_output": "Improved output 2",
            },
        }
    }
    agent_id = "agent_id"
    original_agent = MagicMock()
    original_agent.llm.supports_function_calling.return_value = False
    function_return_value = TrainingTaskEvaluation(
        suggestions=[
            "The initial output was already good, having a detailed explanation. However, the improved output "
            "gave similar information but in a more professional manner using better vocabulary. For future tasks, "
            "try to implement more elaborate language and precise terminology from the beginning."
        ],
        quality=8.0,
        final_summary="The agent responded well initially. However, the improved output showed that there is room "
        "for enhancement in terms of language usage, precision, and professionalism. For future tasks, the agent "
        "should focus more on these points from the start to increase performance.",
    )
    converter_mock.return_value.to_pydantic.return_value = function_return_value
    result = TaskEvaluator(original_agent=original_agent).evaluate_training_data(
        training_data, agent_id
    )

    assert result == function_return_value

    # Verify the converter was called with correct arguments
    converter_mock.assert_called_once()
    call_kwargs = converter_mock.call_args.kwargs

    assert call_kwargs["llm"] == original_agent.llm
    assert call_kwargs["model"] == TrainingTaskEvaluation
    assert "Iteration: data1" in call_kwargs["text"]
    assert "Iteration: data2" in call_kwargs["text"]

    instructions = call_kwargs["instructions"]
    assert "I'm gonna convert this raw text into valid JSON." in instructions
    assert "OpenAPI schema" in instructions
    assert '"type": "json_schema"' in instructions
    assert '"name": "TrainingTaskEvaluation"' in instructions
    assert '"suggestions"' in instructions
    assert '"quality"' in instructions
    assert '"final_summary"' in instructions

    converter_mock.return_value.to_pydantic.assert_called_once()


@patch("crewai.utilities.converter.Converter.to_pydantic")
@patch("crewai.utilities.training_converter.TrainingConverter._convert_field_by_field")
def test_training_converter_fallback_mechanism(
    convert_field_by_field_mock, to_pydantic_mock
):
    training_data = {
        "agent_id": {
            "data1": {
                "initial_output": "Initial output 1",
                "human_feedback": "Human feedback 1",
                "improved_output": "Improved output 1",
            },
            "data2": {
                "initial_output": "Initial output 2",
                "human_feedback": "Human feedback 2",
                "improved_output": "Improved output 2",
            },
        }
    }
    agent_id = "agent_id"
    to_pydantic_mock.side_effect = ConverterError("Failed to convert directly")

    expected_result = TrainingTaskEvaluation(
        suggestions=["Fallback suggestion"],
        quality=6.5,
        final_summary="Fallback summary",
    )
    convert_field_by_field_mock.return_value = expected_result

    original_agent = MagicMock()
    result = TaskEvaluator(original_agent=original_agent).evaluate_training_data(
        training_data, agent_id
    )

    assert result == expected_result
    to_pydantic_mock.assert_called_once()
    convert_field_by_field_mock.assert_called_once()
