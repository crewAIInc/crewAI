import pytest

from crewai.experimental.evaluation.json_parser import extract_json_from_llm_response


def test_extracts_bare_json_object_after_prose() -> None:
    response = 'The evaluation is complete. {"score": 4, "feedback": "clear"}'

    assert extract_json_from_llm_response(response) == {
        "score": 4,
        "feedback": "clear",
    }


def test_extracts_nested_bare_json_object_with_braces_in_string() -> None:
    response = 'Result: {"score": 5, "feedback": "use {value} exactly"} Thanks.'

    assert extract_json_from_llm_response(response)["feedback"] == "use {value} exactly"


def test_rejects_text_without_json() -> None:
    with pytest.raises(ValueError, match="No valid JSON"):
        extract_json_from_llm_response("The judge did not return a score.")
