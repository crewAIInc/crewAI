"""Benchmark runner for NewAgent — run agents against test cases and report results."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    """A single benchmark test case."""

    input: str
    expected: str | None = None
    criteria: str | None = None


class BenchmarkResult(BaseModel):
    """Result of running a single benchmark case."""

    case_index: int
    input: str
    expected: str | None = None
    actual: str = ""
    model: str = ""
    passed: bool = False
    score: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    response_time_ms: int = 0
    cost: float | None = None


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    """Load benchmark cases from a JSON or JSONC file.

    Args:
        path: Path to a JSON/JSONC file containing an array of test cases.

    Returns:
        List of BenchmarkCase instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is not a valid JSON array of cases.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark cases file not found: {path}")

    raw = p.read_text(encoding="utf-8")

    # Strip JSONC comments
    clean = _strip_jsonc_comments(raw)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in benchmark cases file: {e}") from e

    if not isinstance(data, list):
        raise ValueError("Benchmark cases file must contain a JSON array")

    cases: list[BenchmarkCase] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark case at index {i} must be a JSON object")
        if "input" not in item:
            raise ValueError(f"Benchmark case at index {i} missing required 'input' field")
        cases.append(BenchmarkCase(**item))

    return cases


def _strip_jsonc_comments(text: str) -> str:
    """Strip // and /* */ comments from JSONC text."""
    result = re.sub(r"(?<!:)//.*?$", "", text, flags=re.MULTILINE)
    result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)
    return result


def _check_expected(expected: str, actual: str) -> tuple[bool, float]:
    """Check if expected output is found in actual (case-insensitive substring match).

    Returns:
        Tuple of (passed, score).
    """
    if expected.lower() in actual.lower():
        return True, 1.0
    return False, 0.0


async def _judge_with_llm(
    criteria: str,
    input_text: str,
    actual: str,
    judge_model: str,
) -> tuple[bool, float]:
    """Use an LLM judge to evaluate a response against criteria.

    Returns:
        Tuple of (passed, score).
    """
    from crewai.utilities.llm_utils import create_llm

    judge_llm = create_llm(judge_model)

    prompt = (
        "You are an evaluation judge. Score the following response on a scale of 0.0 to 1.0.\n\n"
        f"Input: {input_text}\n\n"
        f"Response: {actual}\n\n"
        f"Evaluation criteria: {criteria}\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"score": <float between 0.0 and 1.0>, "passed": <true or false>}\n'
        "A score >= 0.7 should be considered passed."
    )

    try:
        response = judge_llm.call(messages=[{"role": "user", "content": prompt}])
        text = str(response) if not isinstance(response, str) else response
        # Extract JSON from response
        match = re.search(r"\{[^}]+\}", text)
        if match:
            result = json.loads(match.group())
            score = float(result.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            passed = bool(result.get("passed", score >= 0.7))
            return passed, score
    except Exception:
        pass

    return False, 0.0


def _parse_definition(source: Any) -> dict[str, Any]:
    """Parse an agent definition — delegates to crewai's parser."""
    from crewai.new_agent.definition_parser import parse_agent_definition
    return parse_agent_definition(source)


def _load_agent(source: Any) -> Any:
    """Load a NewAgent from a definition — delegates to crewai's loader."""
    from crewai.new_agent.definition_parser import load_agent_from_definition
    return load_agent_from_definition(source)


async def run_benchmark(
    agent_def: dict[str, Any] | str | Path,
    cases: list[BenchmarkCase],
    models: list[str] | None = None,
    judge_model: str = "openai/gpt-4o-mini",
) -> dict[str, list[BenchmarkResult]]:
    """Run benchmark cases against an agent definition, optionally across multiple models.

    Args:
        agent_def: Agent definition dict, JSON string, or file path.
        cases: List of benchmark cases to run.
        models: Optional list of model identifiers to compare. If None, uses agent's default.
        judge_model: Model to use for LLM judge evaluation.

    Returns:
        Dict mapping model name to list of BenchmarkResult.
    """
    defn = _parse_definition(agent_def)

    if models is None or len(models) == 0:
        models = [defn.get("llm", "default")]

    results_by_model: dict[str, list[BenchmarkResult]] = {}

    for model in models:
        model_results: list[BenchmarkResult] = []

        for i, case in enumerate(cases):
            # Override the model and disable memory for benchmark runs
            bench_defn = dict(defn)
            if model != "default":
                bench_defn["llm"] = model
            bench_defn.setdefault("settings", {})
            bench_defn["settings"]["memory_read_only"] = True

            try:
                agent = _load_agent(bench_defn)
            except Exception as e:
                model_results.append(
                    BenchmarkResult(
                        case_index=i,
                        input=case.input,
                        expected=case.expected,
                        actual=f"[Agent creation error: {e}]",
                        model=model,
                        passed=False,
                        score=0.0,
                    )
                )
                continue

            start_ms = _current_time_ms()
            try:
                response = await agent.amessage(case.input)
                elapsed_ms = _current_time_ms() - start_ms

                actual = response.content
                input_tokens = response.input_tokens or 0
                output_tokens = response.output_tokens or 0
                cost = response.cost

            except Exception as e:
                elapsed_ms = _current_time_ms() - start_ms
                model_results.append(
                    BenchmarkResult(
                        case_index=i,
                        input=case.input,
                        expected=case.expected,
                        actual=f"[Error: {e}]",
                        model=model,
                        passed=False,
                        score=0.0,
                        response_time_ms=elapsed_ms,
                    )
                )
                continue

            # Evaluate
            passed = False
            score = 0.0

            if case.expected is not None:
                passed, score = _check_expected(case.expected, actual)
            if case.criteria is not None:
                criteria_passed, criteria_score = await _judge_with_llm(
                    case.criteria, case.input, actual, judge_model
                )
                if case.expected is not None:
                    # Combine: both must pass, average scores
                    passed = passed and criteria_passed
                    score = (score + criteria_score) / 2.0
                else:
                    passed = criteria_passed
                    score = criteria_score

            model_results.append(
                BenchmarkResult(
                    case_index=i,
                    input=case.input,
                    expected=case.expected,
                    actual=actual,
                    model=model,
                    passed=passed,
                    score=score,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time_ms=elapsed_ms,
                    cost=cost,
                )
            )

        results_by_model[model] = model_results

    return results_by_model


def _current_time_ms() -> int:
    """Return current time in milliseconds."""
    return int(time.monotonic() * 1000)


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a readable table.

    Args:
        results: List of BenchmarkResult for a single model.

    Returns:
        Formatted string table.
    """
    if not results:
        return "No results to display."

    model = results[0].model

    lines: list[str] = []
    lines.append(f"Benchmark Results — Model: {model}")
    lines.append("=" * 80)

    header = f"{'#':<4} {'Pass':<6} {'Score':<7} {'Tokens':<12} {'Time (ms)':<10} {'Input (truncated)'}"
    lines.append(header)
    lines.append("-" * 80)

    total_passed = 0
    total_score = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_time_ms = 0

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        tokens = f"{r.input_tokens}/{r.output_tokens}"
        input_trunc = r.input[:40] + "..." if len(r.input) > 40 else r.input
        line = f"{r.case_index:<4} {status:<6} {r.score:<7.2f} {tokens:<12} {r.response_time_ms:<10} {input_trunc}"
        lines.append(line)

        if r.passed:
            total_passed += 1
        total_score += r.score
        total_input_tokens += r.input_tokens
        total_output_tokens += r.output_tokens
        total_time_ms += r.response_time_ms

    lines.append("-" * 80)
    n = len(results)
    avg_score = total_score / n if n > 0 else 0.0
    lines.append(f"Total: {total_passed}/{n} passed | Avg score: {avg_score:.2f} | "
                 f"Tokens: {total_input_tokens}/{total_output_tokens} | "
                 f"Total time: {total_time_ms}ms")

    return "\n".join(lines)


def format_comparison_table(results_by_model: dict[str, list[BenchmarkResult]]) -> str:
    """Format a comparison table across multiple models.

    Args:
        results_by_model: Dict mapping model name to list of BenchmarkResult.

    Returns:
        Formatted comparison string.
    """
    if not results_by_model:
        return "No results to compare."

    lines: list[str] = []
    lines.append("Model Comparison")
    lines.append("=" * 90)

    header = f"{'Model':<30} {'Passed':<10} {'Avg Score':<12} {'In Tokens':<12} {'Out Tokens':<12} {'Time (ms)'}"
    lines.append(header)
    lines.append("-" * 90)

    for model, results in results_by_model.items():
        n = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / n if n > 0 else 0.0
        total_in = sum(r.input_tokens for r in results)
        total_out = sum(r.output_tokens for r in results)
        total_time = sum(r.response_time_ms for r in results)

        model_trunc = model[:28] if len(model) > 28 else model
        line = (
            f"{model_trunc:<30} {passed}/{n:<8} {avg_score:<12.2f} "
            f"{total_in:<12} {total_out:<12} {total_time}"
        )
        lines.append(line)

    lines.append("-" * 90)

    # Determine best model by average score
    if results_by_model:
        best_model = max(
            results_by_model.keys(),
            key=lambda m: (
                sum(r.score for r in results_by_model[m]) / len(results_by_model[m])
                if results_by_model[m]
                else 0.0
            ),
        )
        best_score = (
            sum(r.score for r in results_by_model[best_model])
            / len(results_by_model[best_model])
            if results_by_model[best_model]
            else 0.0
        )
        lines.append(f"Best model: {best_model} (avg score: {best_score:.2f})")

    return "\n".join(lines)
