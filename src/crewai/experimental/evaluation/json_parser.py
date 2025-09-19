"""Robust JSON parsing utilities for evaluation responses."""

import json
import re
from typing import Any


def extract_json_from_llm_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
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
                return json.loads(match.strip())
            except json.JSONDecodeError:  # noqa: PERF203
                continue
    raise ValueError("No valid JSON found in the response")
