from unittest import mock
from unittest.mock import MagicMock, patch

from crewai.utilities.converter import ConverterError
from crewai.utilities.evaluators.task_evaluator import (
    Entity,
    TaskEvaluation,
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
    converter_mock.assert_has_calls(
        [
            mock.call(
                llm=original_agent.llm,
                text="Assess the quality of the training data based on the llm output, human feedback , and llm "
                "output improved result.\n\nIteration: data1\nInitial Output:\nInitial output 1\n\nHuman Feedback:\nHuman feedback "
                "1\n\nImproved Output:\nImproved output 1\n\n------------------------------------------------\n\nIteration: data2\nInitial Output:\nInitial output 2\n\nHuman "
                "Feedback:\nHuman feedback 2\n\nImproved Output:\nImproved output 2\n\n------------------------------------------------\n\nPlease provide:\n- Provide "
                "a list of clear, actionable instructions derived from the Human Feedbacks to enhance the Agent's "
                "performance. Analyze the differences between Initial Outputs and Improved Outputs to generate specific "
                "action items for future tasks. Ensure all key and specificpoints from the human feedback are "
                "incorporated into these instructions.\n- A score from 0 to 10 evaluating on completion, quality, and "
                "overall performance from the improved output to the initial output based on the human feedback\n",
                model=TrainingTaskEvaluation,
                instructions="I'm gonna convert this raw text into valid JSON.\n\nThe json should have the "
                "following structure, with the following keys:\n{\n    suggestions: List[str],\n    quality: float,\n    final_summary: str\n}",
            ),
            mock.call().to_pydantic(),
        ]
    )


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


def test_task_evaluation_with_missing_quality_field():
    """Test that TaskEvaluation defaults quality to 5.0 when not provided."""
    # Simulate LLM output without quality field
    evaluation_data = {
        "suggestions": ["Test suggestion"],
        "entities": [],
    }

    # Should not raise validation error and should default quality to 5.0
    evaluation = TaskEvaluation(**evaluation_data)

    assert evaluation.quality == 5.0
    assert evaluation.suggestions == ["Test suggestion"]
    assert evaluation.entities == []


def test_task_evaluation_with_provided_quality_field():
    """Test that TaskEvaluation works correctly when quality is provided."""
    # Simulate LLM output with quality field
    evaluation_data = {
        "suggestions": ["Test suggestion"],
        "quality": 8.5,
        "entities": [
            {
                "name": "Test Entity",
                "type": "Person",
                "description": "A test entity",
                "relationships": ["related_to_entity"],
            }
        ],
    }

    evaluation = TaskEvaluation(**evaluation_data)

    assert evaluation.quality == 8.5
    assert evaluation.suggestions == ["Test suggestion"]
    assert len(evaluation.entities) == 1
    assert evaluation.entities[0].name == "Test Entity"


def test_task_evaluation_validation_with_partial_json():
    """Test that TaskEvaluation can be created from partial JSON missing quality."""
    import json

    # Simulate partial JSON response from LLM (missing quality)
    partial_json = json.dumps({
        "suggestions": ["Suggestion 1", "Suggestion 2"],
        "entities": [],
    })

    # Should parse successfully with default quality
    evaluation = TaskEvaluation.model_validate_json(partial_json)

    assert evaluation.quality == 5.0
    assert len(evaluation.suggestions) == 2
    assert evaluation.entities == []
