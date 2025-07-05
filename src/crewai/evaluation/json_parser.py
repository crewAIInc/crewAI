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

    # Try to find any JSON-like structure in the text
    json_object_pattern = r'\{[\s\S]*?\}'
    json_matches = re.findall(json_object_pattern, text, re.DOTALL)
    
    for json_match in json_matches:
        try:
            # Only attempt to parse if it looks substantial enough to be actual JSON
            if len(json_match) > 20:  # Arbitrary threshold to avoid minor matches
                return json.loads(json_match)
        except json.JSONDecodeError:
            continue
            
    # Fallback: Extract structured data from the text
    try:
        # Try to extract scores directly
        scores = {}
        overall_score = 5.0
        
        # Look for overall score
        overall_match = re.search(r'overall[_\s]?score[\s:=]+([0-9](?:\.[0-9]+)?)', 
                                text, re.IGNORECASE)
        if overall_match:
            overall_score = float(overall_match.group(1))
            
        # Try to find scores for individual categories
        categories = ['relevance', 'efficiency', 'parameter_quality', 
                     'strategic_usage', 'result_utilization']
                     
        for category in categories:
            # Try different patterns for each category
            pattern = fr'{category}[\s:=]+([0-9](?:\.[0-9]+)?)'  
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[category] = float(match.group(1))
            else:
                # Default to overall score if we have it, otherwise 5
                scores[category] = overall_score
        
        # Extract feedback
        feedback_match = re.search(r'feedback[\s:=]+(.*?)(?:\n\n|$)', text, re.IGNORECASE | re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else "No detailed feedback available."
        
        # Extract improvement suggestions
        suggestions_match = re.search(r'improvement[\s_]?suggestions[\s:=]+(.*?)(?:\n\n|$)', 
                                   text, re.IGNORECASE | re.DOTALL)
        suggestions = suggestions_match.group(1).strip() if suggestions_match else "No specific suggestions provided."
        
        return {
            "overall_score": overall_score,
            "scores": scores,
            "feedback": feedback,
            "improvement_suggestions": suggestions
        }
        
    except Exception as e:
        logging.warning(f"Failed to extract structured data from text: {e}")
        # Return fallback values
        return {
            "overall_score": 5.0,
            "scores": {
                "relevance": 5.0,
                "efficiency": 5.0,
                "parameter_quality": 5.0,
                "strategic_usage": 5.0,
                "result_utilization": 5.0
            },
            "feedback": "Unable to parse evaluation response.",
            "improvement_suggestions": "Consider providing more structured tool usage patterns."
        }
