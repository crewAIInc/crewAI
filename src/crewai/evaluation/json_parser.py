"""Robust JSON parsing utilities for evaluation responses."""

import json
import re
import logging
from typing import Dict, Any


def extract_json_from_llm_response(text: str) -> Dict[str, Any]:
    """Extract JSON data from text that might contain markdown or other formatting.

    This function uses multiple strategies to extract valid JSON from LLM responses:
    1. Direct JSON parsing
    2. Looking for JSON in code blocks with various patterns
    3. Extracting JSON-like structures
    4. Parsing structured data from the text if no valid JSON is found

    Args:
        text: The text to extract JSON from

    Returns:
        Dict with parsed JSON data or default values
    """
    # First try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Continue with other extraction methods
        pass

    # Look for JSON in code blocks with various patterns
    json_patterns = [
        # Standard markdown code blocks with json
        r'```json\s*([\s\S]*?)\s*```',
        # Code blocks without language specifier
        r'```\s*([\s\S]*?)\s*```',
        # Inline code with JSON
        r'`([{\\[].*[}\]])`',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    return text
