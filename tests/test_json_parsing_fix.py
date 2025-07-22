from pydantic import BaseModel, Field
from crewai.utilities.crew_pydantic_output_parser import clean_json_from_text


class TestOutput(BaseModel):
    summary: str = Field(description="A brief summary")
    confidence: int = Field(description="Confidence level from 1-100")


def test_clean_json_from_text_with_trailing_characters():
    """Test that clean_json_from_text handles trailing characters correctly."""
    text_with_trailing = '''{"summary": "Test summary", "confidence": 85}

Additional text after JSON that should be ignored.
Final Answer: This text should also be ignored.'''
    
    cleaned = clean_json_from_text(text_with_trailing)
    expected = '{"summary": "Test summary", "confidence": 85}'
    assert cleaned == expected


def test_clean_json_from_text_with_markdown():
    """Test that clean_json_from_text handles markdown formatting correctly."""
    text_with_markdown = '''```json
{"summary": "Test summary with markdown", "confidence": 90}
```'''
    
    cleaned = clean_json_from_text(text_with_markdown)
    expected = '{"summary": "Test summary with markdown", "confidence": 90}'
    assert cleaned == expected


def test_clean_json_from_text_with_prefix():
    """Test that clean_json_from_text handles text prefix correctly."""
    text_with_prefix = '''Final Answer: {"summary": "Test summary with prefix", "confidence": 95}'''
    
    cleaned = clean_json_from_text(text_with_prefix)
    expected = '{"summary": "Test summary with prefix", "confidence": 95}'
    assert cleaned == expected


def test_clean_json_from_text_pure_json():
    """Test that clean_json_from_text handles pure JSON correctly."""
    pure_json = '{"summary": "Pure JSON", "confidence": 100}'
    
    cleaned = clean_json_from_text(pure_json)
    assert cleaned == pure_json


def test_clean_json_from_text_no_json():
    """Test that clean_json_from_text returns original text when no JSON found."""
    no_json_text = "This is just plain text with no JSON"
    
    cleaned = clean_json_from_text(no_json_text)
    assert cleaned == no_json_text


def test_clean_json_from_text_invalid_json():
    """Test that clean_json_from_text handles invalid JSON gracefully."""
    invalid_json = '{"summary": "Invalid JSON", "confidence":}'
    
    cleaned = clean_json_from_text(invalid_json)
    assert cleaned == invalid_json


def test_clean_json_from_text_multiple_json_objects():
    """Test that clean_json_from_text returns the first valid JSON object."""
    multiple_json = '''{"summary": "First JSON", "confidence": 80}
    
Some text in between.

{"summary": "Second JSON", "confidence": 90}'''
    
    cleaned = clean_json_from_text(multiple_json)
    expected = '{"summary": "First JSON", "confidence": 80}'
    assert cleaned == expected


def test_clean_json_from_text_nested_json():
    """Test that clean_json_from_text handles nested JSON correctly."""
    nested_json = '''{"summary": "Nested test", "details": {"score": 95, "category": "A"}, "confidence": 85}'''
    
    cleaned = clean_json_from_text(nested_json)
    assert cleaned == nested_json


def test_clean_json_from_text_with_complex_trailing():
    """Test the exact scenario from GitHub issue #3191."""
    github_issue_text = '''{"valid": true, "feedback": null}

Agent failed to reach a final answer. This is likely a bug - please report it.
Error details: maximum recursion depth exceeded in comparison'''
    
    cleaned = clean_json_from_text(github_issue_text)
    expected = '{"valid": true, "feedback": null}'
    assert cleaned == expected
