"""Raised when a completed LLM response cannot be applied to a ``response_model``."""


class StructuredResponseConversionError(Exception):
    """A ``response_model`` could not be applied to an otherwise successful response.

    The provider call itself succeeded, so there is streamed text to return, but that
    text is not the validated payload the caller asked for. Returning it as a result
    would hand back unvalidated data under a validated-model contract, so the
    conversion failure is reported instead.
    """
