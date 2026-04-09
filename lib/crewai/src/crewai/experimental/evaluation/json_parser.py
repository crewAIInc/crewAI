"""Robust JSON parsing utilities for evaluation responses."""

import json
import re
from typing import Any


def extract_json_from_llm_response(text: str) -> dict[str, Any]:
    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError:
        pass

    json_patterns = [
        # Standard markdown code blocks with json
        r"```json\s*([\s\S]*?)\s*```",
        # Code blocks without language specifier
        r"```\s*([\s\S]*?)\s*```",
        # Inline code with JSON
        r"`([{\\[].*[}\]])`",
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                parsed: dict[str, Any] = json.loads(match.strip())
                return parsed
            except json.JSONDecodeError:  # noqa: PERF203
                continue
    raise ValueError("No valid JSON found in the response")
