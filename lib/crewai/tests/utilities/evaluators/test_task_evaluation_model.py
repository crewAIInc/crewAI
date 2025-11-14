"""Tests for TaskEvaluation model validation and normalization.

These tests verify that TaskEvaluation correctly handles malformed LLM output
as reported in issue #3915, including:
- Missing quality field
- Suggestions as list of dicts with 'point' and 'priority' keys
- Score field instead of quality field
- Extra fields that should be ignored
"""

import pytest
from pydantic import ValidationError

from crewai.utilities.evaluators.task_evaluator import Entity, TaskEvaluation


def test_missing_quality_and_dict_suggestions_normalize():
    """Test that missing quality and dict suggestions are normalized correctly.
    
    This replicates the exact error from issue #3915 where:
    - quality field is missing
    - suggestions is a list of dicts with 'point' and 'priority' keys
    """
    payload = {
        "suggestions": [
            {"point": "Proceed immediately with the task", "priority": "high"},
            {"point": "When asking for information, be specific", "priority": "medium"},
            {"point": "Use markdown formatting", "priority": "medium"},
        ],
        "entities": [],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.quality is None
    assert result.suggestions == [
        "Proceed immediately with the task",
        "When asking for information, be specific",
        "Use markdown formatting",
    ]
    assert result.entities == []


def test_single_dict_suggestion():
    """Test that a single dict suggestion is normalized to a list with extracted point."""
    payload = {
        "suggestions": {"point": "Improve response quality", "priority": "high"},
        "quality": 8.0,
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.suggestions == ["Improve response quality"]
    assert result.quality == 8.0


def test_single_string_suggestion():
    """Test that a single string suggestion is normalized to a list."""
    payload = {
        "suggestions": "This is a single suggestion",
        "quality": 7.5,
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.suggestions == ["This is a single suggestion"]
    assert result.quality == 7.5


def test_mixed_suggestions():
    """Test that mixed suggestion types are normalized correctly."""
    payload = {
        "suggestions": [
            "String suggestion",
            {"point": "Dict with point", "priority": "high"},
            {"other": "Dict without point"},
            123,
        ],
        "quality": 6.0,
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.suggestions == [
        "String suggestion",
        "Dict with point",
        "{'other': 'Dict without point'}",
        "123",
    ]


def test_quality_from_score():
    """Test that 'score' field is mapped to 'quality' when quality is missing."""
    payload = {
        "score": 3,
        "suggestions": ["Improve performance"],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.quality == 3.0
    assert result.suggestions == ["Improve performance"]


def test_quality_str_number():
    """Test that quality as a string number is coerced to float."""
    payload = {
        "quality": "7.5",
        "suggestions": ["Good work"],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.quality == 7.5


def test_quality_int():
    """Test that quality as an int is coerced to float."""
    payload = {
        "quality": 8,
        "suggestions": ["Excellent"],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.quality == 8.0


def test_quality_invalid_string():
    """Test that invalid quality string returns None."""
    payload = {
        "quality": "not a number",
        "suggestions": ["Test"],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.quality is None


def test_quality_empty_string():
    """Test that empty quality string returns None."""
    payload = {
        "quality": "",
        "suggestions": ["Test"],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.quality is None


def test_extra_fields_ignored():
    """Test that extra fields in the payload are ignored."""
    payload = {
        "suggestions": ["Test suggestion"],
        "quality": 5.0,
        "relationships": [],
        "extra_field": "should be ignored",
        "another_extra": 123,
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.suggestions == ["Test suggestion"]
    assert result.quality == 5.0
    assert result.entities == []


def test_none_suggestions():
    """Test that None suggestions are normalized to empty list."""
    payload = {
        "suggestions": None,
        "quality": 5.0,
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.suggestions == []


def test_missing_all_optional_fields():
    """Test that all optional fields can be missing."""
    payload = {}
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.suggestions == []
    assert result.quality is None
    assert result.entities == []


def test_entities_with_valid_data():
    """Test that entities are parsed correctly when provided."""
    payload = {
        "suggestions": ["Test"],
        "quality": 8.0,
        "entities": [
            {
                "name": "John Doe",
                "type": "Person",
                "description": "A test person",
                "relationships": ["knows Jane"],
            }
        ],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert len(result.entities) == 1
    assert result.entities[0].name == "John Doe"
    assert result.entities[0].type == "Person"


def test_score_and_quality_both_present():
    """Test that quality takes precedence when both score and quality are present."""
    payload = {
        "score": 3,
        "quality": 9.0,
        "suggestions": ["Test"],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.quality == 9.0


def test_issue_3915_exact_payload():
    """Test the exact payload structure from issue #3915.
    
    This is the actual error case reported in the issue where:
    - quality field is missing
    - suggestions contains dicts with 'point' and 'priority'
    - relationships field is present (should be ignored)
    """
    payload = {
        "score": 3,
        "suggestions": [
            {"point": "Complete the task immediately", "priority": "high"},
            {"point": "Provide detailed explanations", "priority": "medium"},
            {"point": "Use proper formatting", "priority": "medium"},
        ],
        "relationships": ["suggested"],
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.quality == 3.0
    assert result.suggestions == [
        "Complete the task immediately",
        "Provide detailed explanations",
        "Use proper formatting",
    ]
    assert result.entities == []


def test_empty_suggestions_list():
    """Test that empty suggestions list is handled correctly."""
    payload = {
        "suggestions": [],
        "quality": 5.0,
    }
    
    result = TaskEvaluation.model_validate(payload)
    
    assert result.suggestions == []
    assert result.quality == 5.0
