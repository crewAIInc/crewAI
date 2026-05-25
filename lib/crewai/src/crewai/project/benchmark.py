"""Benchmark framework for JSON-defined crewAI agents.

Loads test cases from JSON/JSONC files, runs them against one or more LLM
models via the existing Agent class, and reports pass/fail with scores.
Designed to be invoked by ``crewai test`` on a JSON-configured project.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from crewai.project.json_loader import load_agent, strip_jsonc_comments


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BenchmarkCase(BaseModel):
    """A single benchmark test case."""

    input: str
    expected: str | None = None
    criteria: str | None = None


class BenchmarkResult(BaseModel):
    """Result of running one benchmark case against one model."""

    case_index: int
    input: str
    expected: str | None
    actual: str
    model: str
    passed: bool
    score: float
    input_tokens: int = 0
    output_tokens: int = 0
    response_time_ms: int = 0
    cost: float = 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _judge_with_llm(
    criteria: str,
    input_text: str,
    actual: str,
    judge_model: str,
) -> tuple[bool, float]:
    """Use an LLM to score a response against qualitative *criteria*.

    Returns ``(passed, score)`` where *passed* is ``score >= 0.7``.
    """
    from crewai.llm import LLM

    llm = LLM(model=judge_model)
    prompt = (
        "Evaluate this AI agent response.\n\n"
        f"Input: {input_text}\n"
        f"Response: {actual}\n"
        f"Criteria: {criteria}\n\n"
        "Score from 0.0 to 1.0 where 1.0 is perfect. Return ONLY a JSON object:\n"
        '{"score": 0.8, "passed": true, "reason": "brief explanation"}'
    )
    raw = llm.call([{"role": "user", "content": prompt}])
    return _parse_judge_response(str(raw))


def _parse_judge_response(raw: str) -> tuple[bool, float]:
    """Extract ``(passed, score)`` from a judge LLM response."""
    # Try full JSON parse first.
    try:
        data = json.loads(raw)
        score = float(data["score"])
        passed = bool(data.get("passed", score >= 0.7))
        return passed, max(0.0, min(1.0, score))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Fallback: look for a bare number (e.g. "0.85").
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
    if match:
        score = float(match.group(1))
        return score >= 0.7, max(0.0, min(1.0, score))

    # Cannot parse — conservative fail.
    return False, 0.0


def _score_case(
    case: BenchmarkCase,
    actual: str,
    judge_model: str,
) -> tuple[bool, float]:
    """Score a single benchmark case.

    Scoring rules:
    - No ``expected`` and no ``criteria``: auto-pass, score 1.0.
    - ``expected`` only: case-insensitive substring match → 1.0 / 0.0.
    - ``criteria`` only: LLM judge → score from judge.
    - Both: both must pass; score = average of the two.
    """
    has_expected = case.expected is not None
    has_criteria = case.criteria is not None

    if not has_expected and not has_criteria:
        return True, 1.0

    if has_expected and not has_criteria:
        assert case.expected is not None  # for type-checker
        matched = case.expected.lower() in actual.lower()
        return matched, 1.0 if matched else 0.0

    if has_criteria and not has_expected:
        assert case.criteria is not None
        return _judge_with_llm(case.criteria, case.input, actual, judge_model)

    # Both expected and criteria.
    assert case.expected is not None and case.criteria is not None
    expected_matched = case.expected.lower() in actual.lower()
    expected_score = 1.0 if expected_matched else 0.0
    judge_passed, judge_score = _judge_with_llm(
        case.criteria, case.input, actual, judge_model
    )
    avg_score = (expected_score + judge_score) / 2.0
    return expected_matched and judge_passed, avg_score


# ---------------------------------------------------------------------------
# Benchmark case loader
# ---------------------------------------------------------------------------


def load_benchmark_cases(path: Path) -> tuple[list[BenchmarkCase], float]:
    """Load benchmark cases from a JSON/JSONC file.

    Supports two formats:

    * **Bare array** — ``[{"input": "...", ...}, ...]``
      Threshold defaults to ``0.7``.
    * **Wrapper object** — ``{"threshold": 0.9, "cases": [...]}``
      Uses the provided threshold.

    Returns:
        ``(cases, threshold)``
    """
    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(strip_jsonc_comments(raw_text))

    if isinstance(data, list):
        cases = [BenchmarkCase(**item) for item in data]
        return cases, 0.7

    if isinstance(data, dict):
        threshold = float(data.get("threshold", 0.7))
        raw_cases = data.get("cases", [])
        cases = [BenchmarkCase(**item) for item in raw_cases]
        return cases, threshold

    raise ValueError(
        f"Benchmark file must contain a JSON array or an object with a 'cases' key, got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    agent_path: Path,
    cases: list[BenchmarkCase],
    models: list[str] | None = None,
    judge_model: str = "openai/gpt-4o-mini",
    case_timeout: int = 90,
) -> dict[str, list[BenchmarkResult]]:
    """Run benchmark cases against an agent across one or more models.

    For each model the agent is loaded fresh, its LLM is overridden, and
    memory is disabled for isolation.  Cases within a model are executed
    concurrently (up to 4 at a time) using ``asyncio``.

    Parameters:
        agent_path: Path to the agent's JSON/JSONC definition.
        cases: Pre-loaded benchmark cases.
        models: List of model identifiers to test.  Defaults to the agent's
            own configured LLM.
        judge_model: Model used for qualitative scoring.
        case_timeout: Per-case timeout in seconds.

    Returns:
        Dict mapping each model name to its list of ``BenchmarkResult`` objects.
    """
    if not models:
        probe_agent = load_agent(agent_path)
        default_model = _agent_model_name(probe_agent)
        models = [default_model]

    all_results: dict[str, list[BenchmarkResult]] = {}

    for model in models:
        model_results = _run_model_benchmark(
            agent_path=agent_path,
            cases=cases,
            model=model,
            judge_model=judge_model,
            case_timeout=case_timeout,
        )
        all_results[model] = model_results

    return all_results


def _agent_model_name(agent: Any) -> str:
    """Extract a string model identifier from an agent's LLM."""
    llm = getattr(agent, "llm", None)
    if llm is None:
        return "default"
    if isinstance(llm, str):
        return llm
    return getattr(llm, "model", "default")


def _run_model_benchmark(
    agent_path: Path,
    cases: list[BenchmarkCase],
    model: str,
    judge_model: str,
    case_timeout: int,
) -> list[BenchmarkResult]:
    """Run all cases for a single model, with async concurrency."""

    async def _run_all() -> list[BenchmarkResult]:
        semaphore = asyncio.Semaphore(4)
        tasks = [
            _run_single_case(
                semaphore=semaphore,
                agent_path=agent_path,
                case=case,
                case_index=idx,
                model=model,
                judge_model=judge_model,
                case_timeout=case_timeout,
            )
            for idx, case in enumerate(cases)
        ]
        return list(await asyncio.gather(*tasks))

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        import contextvars

        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(ctx.run, asyncio.run, _run_all()).result()
    else:
        return asyncio.run(_run_all())


async def _run_single_case(
    semaphore: asyncio.Semaphore,
    agent_path: Path,
    case: BenchmarkCase,
    case_index: int,
    model: str,
    judge_model: str,
    case_timeout: int,
) -> BenchmarkResult:
    """Execute and score a single benchmark case (with semaphore gating)."""
    async with semaphore:
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        actual = ""
        try:
            actual = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    _execute_case_sync,
                    agent_path,
                    case,
                    model,
                ),
                timeout=case_timeout,
            )
        except asyncio.TimeoutError:
            actual = f"[TIMEOUT after {case_timeout}s]"
        except Exception as exc:
            actual = f"[ERROR: {exc}]"

        elapsed_ms = int((time.monotonic() - start) * 1000)

        passed, score = _score_case(case, actual, judge_model)

        return BenchmarkResult(
            case_index=case_index,
            input=case.input,
            expected=case.expected,
            actual=actual,
            model=model,
            passed=passed,
            score=score,
            response_time_ms=elapsed_ms,
        )


def _execute_case_sync(
    agent_path: Path,
    case: BenchmarkCase,
    model: str,
) -> str:
    """Load agent, override LLM, and run the case synchronously."""
    from crewai.llm import LLM

    agent = load_agent(agent_path)
    agent.llm = LLM(model=model)
    agent.memory = False
    return agent.message(case.input)


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------


def print_results(
    results: dict[str, list[BenchmarkResult]],
    threshold: float,
) -> None:
    """Print a human-readable summary table of benchmark results.

    Uses plain text only (no Rich dependency).

    Parameters:
        results: Dict mapping model names to result lists.
        threshold: The pass/fail threshold for average score.
    """
    overall_pass = True

    for model, model_results in results.items():
        total = len(model_results)
        passed = sum(1 for r in model_results if r.passed)
        avg_score = (
            sum(r.score for r in model_results) / total if total else 0.0
        )
        avg_time = (
            sum(r.response_time_ms for r in model_results) / total
            if total
            else 0.0
        )
        model_pass = avg_score >= threshold

        if not model_pass:
            overall_pass = False

        # Header
        status = "PASS" if model_pass else "FAIL"
        print(f"\n{'=' * 60}")
        print(f"  Model: {model}")
        print(f"  Status: {status}  (threshold: {threshold:.0%})")
        print(f"  Cases: {passed}/{total} passed")
        print(f"  Avg score: {avg_score:.2f}")
        print(f"  Avg time: {avg_time:.0f}ms")
        print(f"{'=' * 60}")

        # Per-case detail
        print(f"  {'#':<4} {'Pass':<6} {'Score':<7} {'Time':<8} Input")
        print(f"  {'-' * 56}")
        for r in model_results:
            mark = "OK" if r.passed else "FAIL"
            truncated_input = (
                r.input[:40] + "..." if len(r.input) > 40 else r.input
            )
            print(
                f"  {r.case_index:<4} {mark:<6} {r.score:<7.2f} {r.response_time_ms:<8} {truncated_input}"
            )

    # Overall verdict
    print()
    if overall_pass:
        print("BENCHMARK PASSED")
    else:
        print("BENCHMARK FAILED")
        sys.exit(1)
