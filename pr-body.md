## Summary
Add configurable LLM-based retry mechanism when `output_pydantic` / `output_json` validation fails. The `Task` model now exposes `output_validation_max_retries` (default 0 = backward compatible).

## Motivation
Currently, when an agent produces output that fails Pydantic validation, the converter tries `handle_partial_json` once and gives up. This PR adds a retry loop that feeds the validation error back to the LLM so it can self-correct.

## Changes
- **converter.py**: Added `_handle_and_retry`, `_convert_with_retry` helpers (sync + async) with error accumulation across retry attempts. The `max_retries` parameter controls how many LLM retries to attempt.
- **task.py**: Added `output_validation_max_retries` field to `Task`, passed through to `convert_to_model` and `async_convert_to_model`.
- **tests**: 4 new tests in `TestConvertToModelWithRetry` covering success, exhaustion, zero-retry mode, and error context injection.

## Usage
```python
task = Task(
    description="Extract user info",
    expected_output="Structured user data",
    output_pydantic=UserModel,
    output_validation_max_retries=2,
)
```

## Test Plan
- 4 new tests pass
- All 50 existing converter tests pass
- 1 pre-existing failure (test_supports_function_calling_false, missing litellm, unrelated)
- Backward compatible: default max_retries=0 preserves existing behavior
