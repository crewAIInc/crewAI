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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from crewai.project.json_loader import load_agent, strip_jsonc_comments


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BenchmarkCase(BaseModel):
    """A single benchmark test case (agent-level)."""

    input: str
    expected: str | None = None
    criteria: str | None = None


class CrewBenchmarkCase(BaseModel):
    """A benchmark case for crew-level execution."""

    inputs: dict[str, Any] = {}
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
    try:
        data = json.loads(raw)
        score = float(data["score"])
        passed = bool(data.get("passed", score >= 0.7))
        return passed, max(0.0, min(1.0, score))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
    if match:
        score = float(match.group(1))
        return score >= 0.7, max(0.0, min(1.0, score))

    return False, 0.0


def _score_case(
    case: BenchmarkCase,
    actual: str,
    judge_model: str,
) -> tuple[bool, float]:
    """Score a single benchmark case."""
    has_expected = case.expected is not None
    has_criteria = case.criteria is not None

    if not has_expected and not has_criteria:
        return True, 1.0

    if has_expected and not has_criteria:
        assert case.expected is not None
        matched = case.expected.lower() in actual.lower()
        return matched, 1.0 if matched else 0.0

    if has_criteria and not has_expected:
        assert case.criteria is not None
        return _judge_with_llm(case.criteria, case.input, actual, judge_model)

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


def load_crew_benchmark_cases(path: Path) -> list[CrewBenchmarkCase]:
    """Load crew-level benchmark cases from a JSON/JSONC file.

    Each case contains ``inputs`` (dict passed to ``crew.kickoff``) and
    ``criteria``/``expected`` for scoring the result.
    """
    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(strip_jsonc_comments(raw_text))

    if isinstance(data, list):
        return [CrewBenchmarkCase(**item) for item in data]

    if isinstance(data, dict):
        raw_cases = data.get("cases", [])
        return [CrewBenchmarkCase(**item) for item in raw_cases]

    raise ValueError(
        f"Benchmark file must contain a JSON array or object with 'cases' key, got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# Main benchmark runner (agent-level)
# ---------------------------------------------------------------------------


def run_benchmark(
    agent_path: Path,
    cases: list[BenchmarkCase],
    models: list[str] | None = None,
    judge_model: str = "openai/gpt-5.4-mini",
    case_timeout: int = 90,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """Run benchmark cases against an agent across one or more models."""
    if not models:
        probe_agent = load_agent(agent_path)
        default_model = _agent_model_name(probe_agent)
        models = [default_model]

    def _emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    all_results: dict[str, list[BenchmarkResult]] = {}

    for model in models:
        model_results = _run_model_benchmark(
            agent_path=agent_path,
            cases=cases,
            model=model,
            judge_model=judge_model,
            case_timeout=case_timeout,
            emit=_emit,
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
    emit: Callable[[dict[str, Any]], None] = lambda _: None,
) -> list[BenchmarkResult]:
    """Run all cases for a single model, with async concurrency."""
    total = len(cases)
    emit({"type": "model_start", "model": model, "total_cases": total})

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
                emit=emit,
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
            results = pool.submit(ctx.run, asyncio.run, _run_all()).result()
    else:
        results = asyncio.run(_run_all())

    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / total if total else 0.0
    total_time = sum(r.response_time_ms for r in results) / 1000
    emit({
        "type": "model_done",
        "model": model,
        "passed": passed,
        "total": total,
        "avg_score": avg_score,
        "total_time": total_time,
    })
    return results


async def _run_single_case(
    semaphore: asyncio.Semaphore,
    agent_path: Path,
    case: BenchmarkCase,
    case_index: int,
    model: str,
    judge_model: str,
    case_timeout: int,
    emit: Callable[[dict[str, Any]], None] = lambda _: None,
) -> BenchmarkResult:
    """Execute and score a single benchmark case (with semaphore gating)."""
    async with semaphore:
        emit({"type": "case_start", "model": model, "case_index": case_index})
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
        emit({
            "type": "case_done",
            "model": model,
            "case_index": case_index,
            "passed": passed,
            "score": score,
            "time_ms": elapsed_ms,
        })

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
    from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages
    from crewai.llm import LLM

    set_suppress_tracing_messages(True)

    agent = load_agent(agent_path)
    agent.llm = LLM(model=model)
    agent.memory = False
    return agent.message(case.input)


# ---------------------------------------------------------------------------
# Rich output helpers
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


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def print_results(
    results: dict[str, list[BenchmarkResult]],
    threshold: float,
) -> None:
    """Print Rich-formatted benchmark results with score bars."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    overall_pass = True

    for model, model_results in results.items():
        total = len(model_results)
        passed = sum(1 for r in model_results if r.passed)
        avg_score = (
            sum(r.score for r in model_results) / total if total else 0.0
        )
        total_time = sum(r.response_time_ms for r in model_results) / 1000
        model_pass = avg_score >= threshold

        if not model_pass:
            overall_pass = False

        bar_w = 10
        input_w = 35
        rows: list[str] = []
        for r in model_results:
            inp = r.input[: input_w - 1] + "…" if len(r.input) >= input_w else r.input
            inp_pad = inp + " " * max(0, input_w - len(inp))
            bar = _score_bar(r.score, bar_w)
            badge = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            time_s = f"{r.response_time_ms / 1000:>5.1f}s"
            rows.append(
                f"  [dim]{r.case_index + 1:>2}[/dim]  {inp_pad}  {bar} {r.score:.2f}  {badge}  [dim]{time_s}[/dim]"
            )

        color = _score_color(avg_score)
        summary_parts = [
            f"[{color}]{passed}/{total} passed[/{color}]",
            f"avg [{color}]{avg_score:.2f}[/{color}]",
            f"[dim]{total_time:.1f}s[/dim]",
        ]

        body = "\n".join(rows) + "\n\n  " + "  ·  ".join(summary_parts)
        panel = Panel(
            body,
            title=f"[bold cyan]{model}[/bold cyan]",
            subtitle=f"[dim]threshold: {threshold:.0%}[/dim]",
            subtitle_align="left",
            title_align="left",
            border_style="dim",
            padding=(0, 1),
            expand=False,
        )
        console.print()
        console.print(panel)

    console.print()
    if overall_pass:
        console.print(Text("  TEST PASSED", style="bold green"))
    else:
        console.print(Text("  TEST FAILED", style="bold red"))
        sys.exit(1)


def print_results_chart(
    results: list[BenchmarkResult],
    console: Any | None = None,
) -> None:
    """Print Rich panel for a single model's benchmark results (no threshold)."""
    from rich.console import Console
    from rich.panel import Panel

    if not console:
        console = Console()
    if not results:
        return

    model = results[0].model
    bar_w = 10
    input_w = 35

    rows: list[str] = []
    for r in results:
        inp = r.input[: input_w - 1] + "…" if len(r.input) >= input_w else r.input
        inp_pad = inp + " " * max(0, input_w - len(inp))
        bar = _score_bar(r.score, bar_w)
        badge = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        time_s = f"{r.response_time_ms / 1000:>5.1f}s"
        rows.append(
            f"  [dim]{r.case_index + 1:>2}[/dim]  {inp_pad}  {bar} {r.score:.2f}  {badge}  [dim]{time_s}[/dim]"
        )

    n = len(results)
    passed = sum(1 for r in results if r.passed)
    avg = sum(r.score for r in results) / n
    total_time = sum(r.response_time_ms for r in results) / 1000

    color = _score_color(avg)
    summary_parts = [
        f"[{color}]{passed}/{n} passed[/{color}]",
        f"avg [{color}]{avg:.2f}[/{color}]",
        f"[dim]{total_time:.1f}s[/dim]",
    ]

    body = "\n".join(rows) + "\n\n  " + "  ·  ".join(summary_parts)
    panel = Panel(
        body,
        title=f"[bold cyan]{model}[/bold cyan]",
        title_align="left",
        border_style="dim",
        padding=(0, 1),
        expand=False,
    )
    console.print()
    console.print(panel)


def print_comparison_chart(
    results_by_model: dict[str, list[BenchmarkResult]],
    console: Any | None = None,
) -> None:
    """Print Rich panel comparing multiple models side by side."""
    from rich.console import Console
    from rich.panel import Panel

    if not console:
        console = Console()
    if not results_by_model:
        return

    models_data: list[dict[str, Any]] = []
    max_time = 0.0

    for model, results in results_by_model.items():
        n = len(results)
        passed = sum(1 for r in results if r.passed)
        avg = sum(r.score for r in results) / n if n else 0.0
        total_time = sum(r.response_time_ms for r in results) / 1000
        models_data.append({
            "model": model,
            "passed": passed,
            "n": n,
            "avg": avg,
            "time": total_time,
        })
        max_time = max(max_time, total_time)

    for md in models_data:
        time_score = 1.0 - (md["time"] / max_time) if max_time > 0 else 0.0
        md["rank"] = md["avg"] * 0.7 + time_score * 0.3

    best = max(models_data, key=lambda m: m["rank"]) if len(models_data) > 1 else None

    max_name_len = min(max(len(m["model"]) for m in models_data), 28)
    bar_width = 20

    lines: list[str] = []
    for md in models_data:
        name_raw = md["model"]
        name = (
            name_raw[: max_name_len - 1] + "…"
            if len(name_raw) > max_name_len
            else name_raw
        ).ljust(max_name_len)
        bar = _score_bar(md["avg"], bar_width)
        pass_color = _score_color(md["avg"])
        star = (
            " [bold green]★[/bold green]"
            if best and md["model"] == best["model"]
            else ""
        )
        lines.append(
            f"  {name}  {bar} {md['avg']:.2f}  "
            f"[{pass_color}]{md['passed']}/{md['n']}[/{pass_color}]  "
            f"[dim]{md['time']:>5.1f}s[/dim]"
            f"{star}"
        )

    body = "\n".join(lines)
    panel = Panel(
        body,
        title="[bold]Model Comparison[/bold]",
        subtitle="[dim]★ = best (70% score · 30% speed)[/dim]",
        subtitle_align="left",
        title_align="left",
        border_style="dim",
        padding=(1, 1),
        expand=False,
    )
    console.print()
    console.print(panel)


# ---------------------------------------------------------------------------
# Crew-level benchmark runner
# ---------------------------------------------------------------------------


def run_crew_benchmark(
    crew_path: Path,
    cases: list[CrewBenchmarkCase],
    agent: str | None = None,
    models: list[str] | None = None,
    judge_model: str = "openai/gpt-5.4-mini",
    case_timeout: int = 300,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """Run crew-level benchmark, optionally targeting a specific agent.

    When *agent* is given and a suitable checkpoint exists, resumes from
    that checkpoint instead of running the full crew.  When *models* are
    specified alongside *agent*, each model is swapped onto the target
    agent for comparison.
    """

    def _emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    if not models:
        if agent:
            agents_dir = crew_path.parent / "agents"
            for ext in (".jsonc", ".json"):
                af = agents_dir / f"{agent}{ext}"
                if af.exists():
                    probe = load_agent(af)
                    models = [_agent_model_name(probe)]
                    break
        if not models:
            models = ["default"]

    checkpoint_path: Path | None = None
    if agent:
        checkpoint_path = _find_checkpoint_before_agent(crew_path, agent)

    all_results: dict[str, list[BenchmarkResult]] = {}

    for model in models:
        total = len(cases)
        _emit({"type": "model_start", "model": model, "total_cases": total})
        model_results: list[BenchmarkResult] = []

        for idx, case in enumerate(cases):
            _emit({"type": "case_start", "model": model, "case_index": idx})
            start = time.monotonic()
            actual = ""

            try:
                actual = _execute_crew_case_sync(
                    crew_path=crew_path,
                    case=case,
                    agent_name=agent,
                    model_override=model if model != "default" else None,
                    checkpoint_path=checkpoint_path,
                    timeout=case_timeout,
                )
            except Exception as exc:
                actual = f"[ERROR: {exc}]"

            elapsed_ms = int((time.monotonic() - start) * 1000)

            scoring_case = BenchmarkCase(
                input=json.dumps(case.inputs) if case.inputs else "",
                expected=case.expected,
                criteria=case.criteria,
            )
            passed, score = _score_case(scoring_case, actual, judge_model)

            _emit({
                "type": "case_done",
                "model": model,
                "case_index": idx,
                "passed": passed,
                "score": score,
                "time_ms": elapsed_ms,
            })

            model_results.append(BenchmarkResult(
                case_index=idx,
                input=json.dumps(case.inputs) if case.inputs else "(default inputs)",
                expected=case.expected,
                actual=actual,
                model=model,
                passed=passed,
                score=score,
                response_time_ms=elapsed_ms,
            ))

        passed_count = sum(1 for r in model_results if r.passed)
        avg_score = sum(r.score for r in model_results) / total if total else 0.0
        total_time = sum(r.response_time_ms for r in model_results) / 1000
        _emit({
            "type": "model_done",
            "model": model,
            "passed": passed_count,
            "total": total,
            "avg_score": avg_score,
            "total_time": total_time,
        })

        all_results[model] = model_results

    return all_results


def _execute_crew_case_sync(
    crew_path: Path,
    case: CrewBenchmarkCase,
    agent_name: str | None = None,
    model_override: str | None = None,
    checkpoint_path: Path | None = None,
    timeout: int = 300,
) -> str:
    """Execute a single crew benchmark case and return the output string."""
    import logging

    from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages

    set_suppress_tracing_messages(True)
    logging.getLogger("crewai.state.checkpoint_listener").setLevel(logging.CRITICAL)

    if checkpoint_path and agent_name:
        from crewai import Crew
        from crewai.state.checkpoint_config import CheckpointConfig

        config = CheckpointConfig(restore_from=str(checkpoint_path))
        crew = Crew.from_checkpoint(config)
        crew.verbose = False
        for a in crew.agents:
            a.verbose = False

        if model_override:
            _swap_agent_model(crew, crew_path, agent_name, model_override)

        result = crew.kickoff(inputs=case.inputs or {})
    else:
        from crewai.project.crew_loader import load_crew
        from crewai.state.checkpoint_config import CheckpointConfig

        crew, default_inputs = load_crew(crew_path)
        crew.verbose = False
        for a in crew.agents:
            a.verbose = False
        crew.checkpoint = CheckpointConfig()

        if model_override:
            if agent_name:
                _swap_agent_model(crew, crew_path, agent_name, model_override)
            else:
                from crewai.llm import LLM

                for a in crew.agents:
                    a.llm = LLM(model=model_override)

        inputs = {**default_inputs, **(case.inputs or {})}
        result = crew.kickoff(inputs=inputs)

    return str(result.raw) if hasattr(result, "raw") else str(result)


def _swap_agent_model(
    crew: Any, crew_path: Path, agent_name: str, model: str
) -> None:
    """Swap a specific agent's LLM in a loaded crew, matching by role."""
    from crewai.llm import LLM

    agents_dir = crew_path.parent / "agents"
    for ext in (".jsonc", ".json"):
        agent_file = agents_dir / f"{agent_name}{ext}"
        if agent_file.exists():
            probe = load_agent(agent_file)
            for a in crew.agents:
                if a.role == probe.role:
                    a.llm = LLM(model=model)
                    return
            break


def _find_checkpoint_before_agent(
    crew_path: Path,
    agent_name: str,
) -> Path | None:
    """Find the most recent checkpoint where prior tasks are complete but
    the target agent's task has not yet run."""
    raw = crew_path.read_text(encoding="utf-8")
    defn = json.loads(strip_jsonc_comments(raw))
    task_defs = defn.get("tasks", [])

    target_idx = None
    for i, td in enumerate(task_defs):
        if td.get("agent") == agent_name:
            target_idx = i
            break

    if target_idx is None or target_idx == 0:
        return None

    checkpoint_dir = crew_path.parent / ".checkpoints" / "main"
    if not checkpoint_dir.exists():
        return None

    for cp in sorted(checkpoint_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            for entity in data.get("entities", []):
                tasks = entity.get("tasks", [])
                if len(tasks) < target_idx + 1:
                    continue
                before_done = all(
                    t.get("output") is not None for t in tasks[:target_idx]
                )
                target_pending = tasks[target_idx].get("output") is None
                if before_done and target_pending:
                    return cp
        except (json.JSONDecodeError, KeyError):
            continue

    return None
