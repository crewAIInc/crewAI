"""
Constants for OpenTelemetry span attributes.

This module defines constants used for span attributes in telemetry.
"""
from enum import Enum
from typing import Any, Dict


class SpanAttributes:
    """Constants for span attributes used in telemetry."""
    
    OUTPUT_VALUE = "output.value"
    """The output value of an operation."""
    
    INPUT_VALUE = "input.value"
    """The input value of an operation."""
    
    OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
    """The kind of span in OpenInference."""


class OpenInferenceSpanKindValues(Enum):
    """Enum for OpenInference span kind values."""
    
    AGENT = "AGENT"
    CHAIN = "CHAIN"
    LLM = "LLM"
    TOOL = "TOOL"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    UNKNOWN = "UNKNOWN"
    GUARDRAIL = "GUARDRAIL"
    EVALUATOR = "EVALUATOR"
