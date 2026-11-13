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

    # A bare JSON object embedded in prose (no code fence, no inline code),
    # e.g. 'The agent did well.\n{"score": 4, "feedback": "clear"}'. Extract
    # the first balanced {...} region that parses, skipping braces inside
    # strings and non-JSON {...} spans in the surrounding prose.
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            break
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end == -1:
            # This span never closes (stray prose brace), but a later ``{``
            # can still start a complete object — keep scanning instead of
            # giving up on the rest of the response.
            search_from = start + 1
            continue
        try:
            parsed: dict[str, Any] = json.loads(text[start : end + 1])
            return parsed
        except json.JSONDecodeError:
            search_from = start + 1  # not JSON; try the next {...} span

    raise ValueError("No valid JSON found in the response")
