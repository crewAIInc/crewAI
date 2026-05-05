"""Derive a run outcome from observable signals.

No human grading required. The reviewer treats outcomes as
confidence-weighted hints, not ground truth — clustering across runs
absorbs label noise.
"""

from __future__ import annotations

from collections import Counter

from crewai.skills.self_improve.models import Outcome, RunTrace


_RETRY_THRASH_THRESHOLD = 3


def _has_thrashing(trace: RunTrace) -> bool:
    """Detect a tool being called repeatedly with the same args summary.

    A tool fired more than ``_RETRY_THRASH_THRESHOLD`` times with identical
    args is likely stuck in a retry loop rather than making progress.
    """
    if len(trace.tool_calls) <= _RETRY_THRASH_THRESHOLD:
        return False
    keys = [(t.name, t.args_summary) for t in trace.tool_calls]
    most_common_count = Counter(keys).most_common(1)[0][1]
    return most_common_count > _RETRY_THRASH_THRESHOLD


def _output_looks_empty(trace: RunTrace) -> bool:
    if trace.output_summary is None:
        return False
    stripped = trace.output_summary.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    return lowered.startswith("error") or lowered.startswith("traceback")


def grade_trace(trace: RunTrace) -> Outcome:
    """Compute outcome from signals already present on the trace.

    Signal hierarchy (strongest first):
        1. explicit error → failure
        2. guardrail decided → trust it
        3. max_iter exhaustion → failure
        4. tool error rate / thrashing / empty output → failure or partial
        5. otherwise → success when we saw output, else unknown
    """
    if trace.error:
        return "failure"

    if trace.guardrail_passed is True:
        return "success"
    if trace.guardrail_passed is False:
        return "failure"

    if trace.max_iter_exhausted:
        return "failure"

    if _has_thrashing(trace):
        return "failure"

    if _output_looks_empty(trace):
        return "failure"

    if trace.tool_call_count > 0:
        error_rate = trace.tool_error_count / trace.tool_call_count
        if error_rate >= 0.5:
            return "failure"

    if trace.output_summary:
        return "success"

    return "unknown"
