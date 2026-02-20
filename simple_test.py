#!/usr/bin/env python3
"""Simple test for the markdown JSON fix."""

import json
import re

# Copy the function directly for testing
def strip_markdown_code_blocks(text: str) -> str:
    """Strip markdown code blocks from text to extract raw JSON."""
    markdown_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    match = markdown_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text

def test_fix():
    print("Testing markdown JSON parsing fix...")
    
    # The problematic input that was causing the error
    problematic_input = '''```json
{
  "suggestions": [
    "Include more specific metrics and data points",
    "Consider adding visual charts or graphs",
    "Expand on potential challenges"
  ],
  "quality": 8.5,
  "entities": [
    {
      "name": "AI in Healthcare",
      "type": "Technology Domain",
      "description": "Application of AI in medical settings",
      "relationships": ["Medical Diagnosis", "Patient Care"]
    }
  ]
}
```'''
    
    print("1. Testing markdown stripping...")
    cleaned = strip_markdown_code_blocks(problematic_input)
    print("✓ Stripped markdown blocks")
    
    print("2. Testing JSON parsing...")
    try:
        parsed = json.loads(cleaned)
        print("✓ Successfully parsed JSON")
        print(f"   Quality: {parsed['quality']}")
        print(f"   Suggestions count: {len(parsed['suggestions'])}")
        print(f"   Entities count: {len(parsed['entities'])}")
    except Exception as e:
        print(f"✗ Failed to parse: {e}")
        return False
    
    print("3. Testing edge cases...")
    
    # Regular JSON should remain unchanged
    regular_json = '{"test": "value"}'
    if strip_markdown_code_blocks(regular_json) == regular_json:
        print("✓ Regular JSON unchanged")
    else:
        print("✗ Regular JSON was modified")
        return False
    
    # Code blocks without language specifier
    no_lang = '''```
{"test": "value"}
```'''
    try:
        json.loads(strip_markdown_code_blocks(no_lang))
        print("✓ No language specifier works")
    except:
        print("✗ No language specifier failed")
        return False
    
    print("\n🎉 All tests passed! The fix should work.")
    return True

if __name__ == "__main__":
    test_fix()