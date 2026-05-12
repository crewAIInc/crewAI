"""Benchmark runner for NewAgent — run agents against test cases and report results."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

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
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """Run benchmark cases against an agent definition, optionally across multiple models.

    Args:
        agent_def: Agent definition dict, JSON string, or file path.
        cases: List of benchmark cases to run.
        models: Optional list of model identifiers to compare. If None, uses agent's default.
        judge_model: Model to use for LLM judge evaluation.
        on_progress: Optional callback receiving progress dicts with a "type" key.

    Returns:
        Dict mapping model name to list of BenchmarkResult.
    """
    defn = _parse_definition(agent_def)

    if models is None or len(models) == 0:
        models = [defn.get("llm", "default")]

    def _emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    results_by_model: dict[str, list[BenchmarkResult]] = {}

    for mi, model in enumerate(models):
        model_results: list[BenchmarkResult] = []
        _emit({"type": "model_start", "model": model, "model_index": mi, "total_models": len(models), "total_cases": len(cases)})

        for i, case in enumerate(cases):
            _emit({"type": "case_start", "model": model, "case_index": i, "total_cases": len(cases), "input": case.input})

            bench_defn = dict(defn)
            if model != "default":
                bench_defn["llm"] = model
            bench_defn.setdefault("settings", {})
            bench_defn["settings"]["memory_read_only"] = True

            try:
                agent = _load_agent(bench_defn)
            except Exception as e:
                result = BenchmarkResult(
                    case_index=i,
                    input=case.input,
                    expected=case.expected,
                    actual=f"[Agent creation error: {e}]",
                    model=model,
                    passed=False,
                    score=0.0,
                )
                model_results.append(result)
                _emit({"type": "case_done", "model": model, "case_index": i, "total_cases": len(cases), "passed": False, "score": 0.0, "time_ms": 0, "error": str(e)})
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
                result = BenchmarkResult(
                    case_index=i,
                    input=case.input,
                    expected=case.expected,
                    actual=f"[Error: {e}]",
                    model=model,
                    passed=False,
                    score=0.0,
                    response_time_ms=elapsed_ms,
                )
                model_results.append(result)
                _emit({"type": "case_done", "model": model, "case_index": i, "total_cases": len(cases), "passed": False, "score": 0.0, "time_ms": elapsed_ms, "error": str(e)})
                continue

            passed = False
            score = 0.0

            if case.expected is not None:
                passed, score = _check_expected(case.expected, actual)
            if case.criteria is not None:
                _emit({"type": "judging", "model": model, "case_index": i, "total_cases": len(cases)})
                criteria_passed, criteria_score = await _judge_with_llm(
                    case.criteria, case.input, actual, judge_model
                )
                if case.expected is not None:
                    passed = passed and criteria_passed
                    score = (score + criteria_score) / 2.0
                else:
                    passed = criteria_passed
                    score = criteria_score

            result = BenchmarkResult(
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
            model_results.append(result)
            _emit({"type": "case_done", "model": model, "case_index": i, "total_cases": len(cases), "passed": passed, "score": score, "time_ms": elapsed_ms})

        results_by_model[model] = model_results
        total_passed = sum(1 for r in model_results if r.passed)
        avg_score = sum(r.score for r in model_results) / len(model_results) if model_results else 0.0
        _emit({"type": "model_done", "model": model, "passed": total_passed, "total": len(model_results), "avg_score": avg_score})

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


# ---------------------------------------------------------------------------
# Rich-based terminal charts
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "yellow"
    return "red"


def _score_bar(score: float, width: int = 20) -> str:
    clamped = max(0.0, min(1.0, score))
    filled = round(clamped * width)
    empty = width - filled
    color = _score_color(score)
    bar = f"[{color}]{'█' * filled}[/{color}]"
    if empty:
        bar += f"[dim]{'░' * empty}[/dim]"
    return bar


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_cost(cost: float | None) -> str:
    if cost is None:
        return ""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def print_results_chart(
    results: list[BenchmarkResult],
    console: Any | None = None,
) -> None:
    from rich.console import Console
    from rich.panel import Panel

    if not console:
        console = Console()

    if not results:
        console.print("[dim]No results to display.[/]")
        return

    model = results[0].model
    has_cost = any(r.cost is not None for r in results)

    inner_w = max(console.width - 4, 60)
    bar_w = 12
    overhead = 2 + 2 + 2 + 2 + bar_w + 1 + 4 + 2 + 4 + 2 + 6
    if has_cost:
        overhead += 2 + 7
    input_w = max(15, inner_w - overhead)

    rows: list[str] = []
    for r in results:
        inp = r.input[:input_w - 1] + "…" if len(r.input) >= input_w else r.input
        inp_pad = inp + " " * max(0, input_w - len(inp))
        bar = _score_bar(r.score, bar_w)
        badge = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        time_s = f"{r.response_time_ms / 1000:>5.1f}s"
        cost_part = f"  [dim]{_fmt_cost(r.cost):>7}[/dim]" if has_cost else ""
        rows.append(
            f"  [dim]{r.case_index:>2}[/dim]  {inp_pad}  {bar} {r.score:.2f}  {badge}  [dim]{time_s}[/dim]{cost_part}"
        )

    n = len(results)
    passed = sum(1 for r in results if r.passed)
    avg = sum(r.score for r in results) / n
    total_time = sum(r.response_time_ms for r in results) / 1000
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    total_cost = sum(r.cost for r in results if r.cost is not None)

    color = _score_color(avg)
    summary_parts = [
        f"[{color}]{passed}/{n} passed[/]",
        f"avg [{color}]{avg:.2f}[/]",
        f"[dim]{total_time:.1f}s[/]",
        f"[dim]↑{_fmt_tokens(total_in)} ↓{_fmt_tokens(total_out)}[/]",
    ]
    if total_cost > 0:
        summary_parts.append(f"[dim]{_fmt_cost(total_cost)}[/]")

    body = "\n".join(rows) + "\n\n  " + "  ·  ".join(summary_parts)
    panel = Panel(
        body,
        title=f"[bold cyan]{model}[/]",
        title_align="left",
        border_style="cyan",
        padding=(1, 0),
        expand=False,
    )
    console.print(panel)


def print_comparison_chart(
    results_by_model: dict[str, list[BenchmarkResult]],
    console: Any | None = None,
) -> None:
    from rich.console import Console
    from rich.panel import Panel

    if not console:
        console = Console()

    if not results_by_model:
        console.print("[dim]No results to compare.[/]")
        return

    inner_w = max(console.width - 4, 60)
    fixed_right = 1 + 4 + 2 + 5 + 2 + 6 + 4
    models_data: list[tuple[str, int, int, float, float]] = []
    best_model = ""
    best_score = -1.0

    for model, results in results_by_model.items():
        n = len(results)
        passed = sum(1 for r in results if r.passed)
        avg = sum(r.score for r in results) / n if n else 0.0
        total_time = sum(r.response_time_ms for r in results) / 1000
        models_data.append((model, passed, n, avg, total_time))
        if avg > best_score:
            best_score = avg
            best_model = model

    max_name_len = min(max(len(m) for m, *_ in models_data), 28)
    bar_width = max(12, inner_w - max_name_len - fixed_right - 4)
    bar_width = min(bar_width, 30)

    lines: list[str] = []
    for model, passed, n, avg, total_time in models_data:
        name = (model[:max_name_len - 1] + "…" if len(model) > max_name_len else model).ljust(max_name_len)
        bar = _score_bar(avg, bar_width)
        pass_color = _score_color(avg)
        star = " [bold green]★[/]" if model == best_model and len(models_data) > 1 else ""
        lines.append(
            f"  {name}  {bar} {avg:.2f}  "
            f"[{pass_color}]{passed}/{n}[/]  "
            f"[dim]{total_time:>5.1f}s[/]"
            f"{star}"
        )

    body = "\n".join(lines)
    panel = Panel(
        body,
        title="[bold]Model Comparison[/]",
        title_align="left",
        border_style="dim",
        padding=(1, 1),
        expand=False,
    )
    console.print(panel)
