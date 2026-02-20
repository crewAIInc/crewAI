#!/usr/bin/env python3
"""Test script to verify the markdown JSON parsing fix."""

import json
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib/crewai/src'))

from crewai.utilities.converter import strip_markdown_code_blocks
from pydantic import BaseModel, Field
from typing import List


class Entity(BaseModel):
    name: str = Field(description="The name of the entity.")
    type: str = Field(description="The type of the entity.")
    description: str = Field(description="Description of the entity.")
    relationships: List[str] = Field(description="Relationships of the entity.")


class TaskEvaluation(BaseModel):
    suggestions: List[str] = Field(
        description="Suggestions to improve future similar tasks."
    )
    quality: float = Field(
        description="A score from 0 to 10 evaluating on completion, quality, and overall performance."
    )
    entities: List[Entity] = Field(
        description="Entities extracted from the task output."
    )


def test_markdown_json_parsing():
    """Test that markdown JSON blocks are correctly parsed."""
    print("Testing markdown JSON parsing fix...")
    
    # Simulate the problematic LLM response that was failing
    markdown_wrapped_json = '''```json
{
  "suggestions": [
    "Include more specific metrics and data points to support findings",
    "Consider adding visual charts or graphs for better data presentation",
    "Expand on the potential challenges and limitations of AI in healthcare"
  ],
  "quality": 8.5,
  "entities": [
    {
      "name": "AI in Healthcare",
      "type": "Technology Domain",
      "description": "Application of artificial intelligence technologies in medical and healthcare settings",
      "relationships": ["Medical Diagnosis", "Patient Care", "Healthcare Efficiency"]
    },
    {
      "name": "Machine Learning",
      "type": "Technology",
      "description": "Subset of AI used for pattern recognition and predictive analytics in healthcare",
      "relationships": ["Medical Imaging", "Drug Discovery", "Predictive Analytics"]
    }
  ]
}
```'''

    # Test the strip function
    print("1. Testing strip_markdown_code_blocks function...")
    cleaned_json = strip_markdown_code_blocks(markdown_wrapped_json)
    print("✓ Successfully stripped markdown blocks")
    
    # Test that the cleaned JSON can be parsed
    print("2. Testing JSON parsing...")
    try:
        parsed_data = json.loads(cleaned_json)
        print("✓ Successfully parsed cleaned JSON")
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse cleaned JSON: {e}")
        return False
    
    # Test that Pydantic validation works
    print("3. Testing Pydantic validation...")
    try:
        task_eval = TaskEvaluation.model_validate_json(cleaned_json)
        print("✓ Successfully validated with Pydantic")
        print(f"  - Quality score: {task_eval.quality}")
        print(f"  - Number of suggestions: {len(task_eval.suggestions)}")
        print(f"  - Number of entities: {len(task_eval.entities)}")
    except Exception as e:
        print(f"✗ Failed Pydantic validation: {e}")
        return False
    
    # Test edge cases
    print("4. Testing edge cases...")
    
    # Test regular JSON (no markdown)
    regular_json = '{"suggestions": ["test"], "quality": 9.0, "entities": []}'
    cleaned_regular = strip_markdown_code_blocks(regular_json)
    if cleaned_regular == regular_json:
        print("✓ Regular JSON unchanged")
    else:
        print("✗ Regular JSON was modified incorrectly")
        return False
    
    # Test JSON with 'json' language specifier
    json_with_lang = '''```json
{"suggestions": ["test"], "quality": 7.0, "entities": []}
```'''
    cleaned_lang = strip_markdown_code_blocks(json_with_lang)
    try:
        json.loads(cleaned_lang)
        print("✓ JSON with language specifier cleaned correctly")
    except:
        print("✗ Failed to clean JSON with language specifier")
        return False
        
    # Test JSON without language specifier
    json_no_lang = '''```
{"suggestions": ["test"], "quality": 6.0, "entities": []}
```'''
    cleaned_no_lang = strip_markdown_code_blocks(json_no_lang)
    try:
        json.loads(cleaned_no_lang)
        print("✓ JSON without language specifier cleaned correctly")
    except:
        print("✗ Failed to clean JSON without language specifier")
        return False
    
    print("\n🎉 All tests passed! The fix should resolve the Pydantic validation error.")
    return True


if __name__ == "__main__":
    success = test_markdown_json_parsing()
    sys.exit(0 if success else 1)