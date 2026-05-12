"""Tests for the benchmark module."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai_cli.benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    _check_expected,
    _strip_jsonc_comments,
    format_comparison_table,
    format_results_table,
    load_benchmark_cases,
    run_benchmark,
)


# ── BenchmarkCase model tests ──────────────────────────────────


class TestBenchmarkCase:
    def test_with_expected(self):
        case = BenchmarkCase(input="What is 2+2?", expected="4")
        assert case.input == "What is 2+2?"
        assert case.expected == "4"
        assert case.criteria is None

    def test_with_criteria(self):
        case = BenchmarkCase(
            input="Write a haiku",
            criteria="Must be a valid haiku",
        )
        assert case.input == "Write a haiku"
        assert case.expected is None
        assert case.criteria == "Must be a valid haiku"

    def test_with_both(self):
        case = BenchmarkCase(
            input="Answer", expected="42", criteria="Must be numeric"
        )
        assert case.expected == "42"
        assert case.criteria == "Must be numeric"

    def test_input_only(self):
        case = BenchmarkCase(input="Hello")
        assert case.expected is None
        assert case.criteria is None


# ── BenchmarkResult model tests ──────────────────────────────────


class TestBenchmarkResult:
    def test_defaults(self):
        r = BenchmarkResult(case_index=0, input="test")
        assert r.case_index == 0
        assert r.input == "test"
        assert r.passed is False
        assert r.score == 0.0
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.response_time_ms == 0
        assert r.cost is None
        assert r.model == ""
        assert r.actual == ""

    def test_full(self):
        r = BenchmarkResult(
            case_index=1,
            input="What is 2+2?",
            expected="4",
            actual="The answer is 4",
            model="openai/gpt-4o",
            passed=True,
            score=1.0,
            input_tokens=50,
            output_tokens=10,
            response_time_ms=500,
            cost=0.001,
        )
        assert r.passed is True
        assert r.cost == 0.001


# ── load_benchmark_cases tests ──────────────────────────────────


class TestLoadBenchmarkCases:
    def test_load_json(self, tmp_path: Path):
        cases_data = [
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "Write a haiku", "criteria": "Must be 5-7-5"},
        ]
        f = tmp_path / "cases.json"
        f.write_text(json.dumps(cases_data), encoding="utf-8")

        cases = load_benchmark_cases(f)
        assert len(cases) == 2
        assert cases[0].input == "What is 2+2?"
        assert cases[0].expected == "4"
        assert cases[1].criteria == "Must be 5-7-5"

    def test_load_jsonc(self, tmp_path: Path):
        jsonc_content = """[
  // A simple math test
  {"input": "What is 2+2?", "expected": "4"},
  /* Multi-line
     comment */
  {"input": "Hello", "criteria": "Must be polite"}
]"""
        f = tmp_path / "cases.jsonc"
        f.write_text(jsonc_content, encoding="utf-8")

        cases = load_benchmark_cases(f)
        assert len(cases) == 2
        assert cases[0].expected == "4"
        assert cases[1].criteria == "Must be polite"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_benchmark_cases("/nonexistent/path.json")

    def test_invalid_json(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_benchmark_cases(f)

    def test_not_array(self, tmp_path: Path):
        f = tmp_path / "obj.json"
        f.write_text('{"input": "test"}', encoding="utf-8")

        with pytest.raises(ValueError, match="must contain a JSON array"):
            load_benchmark_cases(f)

    def test_missing_input_field(self, tmp_path: Path):
        f = tmp_path / "missing.json"
        f.write_text('[{"expected": "4"}]', encoding="utf-8")

        with pytest.raises(ValueError, match="missing required 'input' field"):
            load_benchmark_cases(f)

    def test_non_object_item(self, tmp_path: Path):
        f = tmp_path / "bad_items.json"
        f.write_text('["not an object"]', encoding="utf-8")

        with pytest.raises(ValueError, match="must be a JSON object"):
            load_benchmark_cases(f)

    def test_string_path(self, tmp_path: Path):
        cases_data = [{"input": "Hello"}]
        f = tmp_path / "str_path.json"
        f.write_text(json.dumps(cases_data), encoding="utf-8")

        cases = load_benchmark_cases(str(f))
        assert len(cases) == 1


# ── _strip_jsonc_comments tests ──────────────────────────────────


class TestStripJsoncComments:
    def test_no_comments(self):
        text = '{"key": "value"}'
        assert json.loads(_strip_jsonc_comments(text)) == {"key": "value"}

    def test_single_line_comments(self):
        text = '{\n  // comment\n  "key": "value"\n}'
        result = json.loads(_strip_jsonc_comments(text))
        assert result == {"key": "value"}

    def test_multi_line_comments(self):
        text = '{\n  /* multi\n  line */\n  "key": "value"\n}'
        result = json.loads(_strip_jsonc_comments(text))
        assert result == {"key": "value"}


# ── _check_expected tests ──────────────────────────────────


class TestCheckExpected:
    def test_exact_match(self):
        passed, score = _check_expected("4", "4")
        assert passed is True
        assert score == 1.0

    def test_substring_match(self):
        passed, score = _check_expected("4", "The answer is 4.")
        assert passed is True
        assert score == 1.0

    def test_case_insensitive(self):
        passed, score = _check_expected("hello", "HELLO WORLD")
        assert passed is True
        assert score == 1.0

    def test_no_match(self):
        passed, score = _check_expected("banana", "The answer is apple")
        assert passed is False
        assert score == 0.0


# ── format_results_table tests ──────────────────────────────────


class TestFormatResultsTable:
    def test_empty_results(self):
        output = format_results_table([])
        assert output == "No results to display."

    def test_single_result(self):
        results = [
            BenchmarkResult(
                case_index=0,
                input="What is 2+2?",
                expected="4",
                actual="4",
                model="openai/gpt-4o",
                passed=True,
                score=1.0,
                input_tokens=50,
                output_tokens=10,
                response_time_ms=200,
            )
        ]
        output = format_results_table(results)
        assert "openai/gpt-4o" in output
        assert "PASS" in output
        assert "1/1 passed" in output
        assert "Avg score: 1.00" in output

    def test_multiple_results_mixed(self):
        results = [
            BenchmarkResult(
                case_index=0,
                input="Q1",
                model="m1",
                passed=True,
                score=1.0,
                input_tokens=10,
                output_tokens=5,
                response_time_ms=100,
            ),
            BenchmarkResult(
                case_index=1,
                input="Q2",
                model="m1",
                passed=False,
                score=0.3,
                input_tokens=20,
                output_tokens=8,
                response_time_ms=150,
            ),
        ]
        output = format_results_table(results)
        assert "1/2 passed" in output
        assert "PASS" in output
        assert "FAIL" in output
        # Avg score = (1.0 + 0.3) / 2 = 0.65
        assert "0.65" in output

    def test_long_input_truncated(self):
        long_input = "A" * 100
        results = [
            BenchmarkResult(
                case_index=0,
                input=long_input,
                model="m1",
                passed=True,
                score=1.0,
            )
        ]
        output = format_results_table(results)
        assert "..." in output


# ── format_comparison_table tests ──────────────────────────────────


class TestFormatComparisonTable:
    def test_empty(self):
        output = format_comparison_table({})
        assert output == "No results to compare."

    def test_single_model(self):
        results_by_model = {
            "openai/gpt-4o": [
                BenchmarkResult(
                    case_index=0,
                    input="Q1",
                    model="openai/gpt-4o",
                    passed=True,
                    score=1.0,
                    input_tokens=50,
                    output_tokens=10,
                    response_time_ms=200,
                )
            ]
        }
        output = format_comparison_table(results_by_model)
        assert "openai/gpt-4o" in output
        assert "Best model: openai/gpt-4o" in output

    def test_multi_model_comparison(self):
        results_by_model = {
            "model-a": [
                BenchmarkResult(
                    case_index=0, input="Q1", model="model-a",
                    passed=True, score=0.9, input_tokens=50,
                    output_tokens=10, response_time_ms=200,
                ),
                BenchmarkResult(
                    case_index=1, input="Q2", model="model-a",
                    passed=True, score=0.8, input_tokens=60,
                    output_tokens=15, response_time_ms=300,
                ),
            ],
            "model-b": [
                BenchmarkResult(
                    case_index=0, input="Q1", model="model-b",
                    passed=False, score=0.3, input_tokens=40,
                    output_tokens=8, response_time_ms=150,
                ),
                BenchmarkResult(
                    case_index=1, input="Q2", model="model-b",
                    passed=False, score=0.2, input_tokens=45,
                    output_tokens=12, response_time_ms=250,
                ),
            ],
        }
        output = format_comparison_table(results_by_model)
        assert "model-a" in output
        assert "model-b" in output
        assert "Best model: model-a" in output
        assert "Model Comparison" in output


# ── run_benchmark tests (mocked LLM) ──────────────────────────────────


def _make_mock_agent(content: str = "The answer is 4", input_tokens: int = 50, output_tokens: int = 10):
    """Create a mock agent that returns a fixed response."""
    from crewai.new_agent.models import Message

    mock_response = Message(
        role="agent",
        content=content,
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_time_ms=100,
    )

    mock_agent = MagicMock()
    mock_agent.amessage = AsyncMock(return_value=mock_response)
    return mock_agent


class TestRunBenchmark:
    def test_single_case_expected_pass(self):
        cases = [BenchmarkCase(input="What is 2+2?", expected="4")]
        mock_agent = _make_mock_agent("The answer is 4")

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "test-model"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
            ))

        assert "test-model" in results
        assert len(results["test-model"]) == 1
        assert results["test-model"][0].passed is True
        assert results["test-model"][0].score == 1.0

    def test_single_case_expected_fail(self):
        cases = [BenchmarkCase(input="What is 2+2?", expected="banana")]
        mock_agent = _make_mock_agent("The answer is 4")

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "test-model"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
            ))

        assert results["test-model"][0].passed is False
        assert results["test-model"][0].score == 0.0

    def test_multiple_cases(self):
        cases = [
            BenchmarkCase(input="Q1", expected="4"),
            BenchmarkCase(input="Q2", expected="banana"),
        ]
        mock_agent = _make_mock_agent("The answer is 4")

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "test-model"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
            ))

        assert len(results["test-model"]) == 2
        assert results["test-model"][0].passed is True
        assert results["test-model"][1].passed is False

    def test_multi_model_comparison(self):
        cases = [BenchmarkCase(input="Q1", expected="4")]
        mock_agent = _make_mock_agent("The answer is 4")

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "default"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
                models=["model-a", "model-b"],
            ))

        assert "model-a" in results
        assert "model-b" in results
        assert len(results["model-a"]) == 1
        assert len(results["model-b"]) == 1

    def test_criteria_evaluation(self):
        cases = [BenchmarkCase(input="Write a haiku", criteria="Must be a valid haiku")]
        mock_agent = _make_mock_agent("Old pond / frog leaps in / water's sound")

        mock_judge_result = (True, 0.9)

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "test-model"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent), \
             patch("crewai_cli.benchmark._judge_with_llm", new_callable=AsyncMock, return_value=mock_judge_result):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
            ))

        assert results["test-model"][0].passed is True
        assert results["test-model"][0].score == 0.9

    def test_combined_expected_and_criteria(self):
        cases = [
            BenchmarkCase(
                input="What is 2+2?",
                expected="4",
                criteria="Must be numeric",
            )
        ]
        mock_agent = _make_mock_agent("The answer is 4")
        mock_judge_result = (True, 0.8)

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "test-model"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent), \
             patch("crewai_cli.benchmark._judge_with_llm", new_callable=AsyncMock, return_value=mock_judge_result):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
            ))

        r = results["test-model"][0]
        assert r.passed is True
        # Score should be average of expected (1.0) and criteria (0.8) = 0.9
        assert r.score == pytest.approx(0.9)

    def test_agent_creation_error(self):
        cases = [BenchmarkCase(input="Q1", expected="4")]

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "test-model"}), \
             patch("crewai_cli.benchmark._load_agent", side_effect=Exception("Agent init failed")):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
            ))

        r = results["test-model"][0]
        assert r.passed is False
        assert "Agent creation error" in r.actual

    def test_agent_message_error(self):
        cases = [BenchmarkCase(input="Q1", expected="4")]
        mock_agent = MagicMock()
        mock_agent.amessage = AsyncMock(side_effect=Exception("LLM timeout"))

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "test-model"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
            ))

        r = results["test-model"][0]
        assert r.passed is False
        assert "Error" in r.actual

    def test_tokens_and_timing_recorded(self):
        cases = [BenchmarkCase(input="Q1", expected="4")]
        mock_agent = _make_mock_agent("4", input_tokens=100, output_tokens=25)

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "test-model"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
            ))

        r = results["test-model"][0]
        assert r.input_tokens == 100
        assert r.output_tokens == 25
        assert r.response_time_ms >= 0

    def test_default_model_used(self):
        """When no models specified, uses agent's default llm."""
        cases = [BenchmarkCase(input="Q1", expected="4")]
        mock_agent = _make_mock_agent("4")

        with patch("crewai_cli.benchmark._parse_definition", return_value={"role": "test", "goal": "test", "llm": "openai/gpt-4o"}), \
             patch("crewai_cli.benchmark._load_agent", return_value=mock_agent):
            results = asyncio.run(run_benchmark(
                agent_def={"role": "test", "goal": "test"},
                cases=cases,
                models=None,
            ))

        assert "openai/gpt-4o" in results
