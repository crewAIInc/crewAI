"""Tests for crewai.project.benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from crewai.project.benchmark import (
    BenchmarkCase,
    CrewBenchmarkCase,
    _find_checkpoint_before_agent,
    _parse_judge_response,
    _score_case,
    _execute_crew_case_with_timeout,
    load_benchmark_cases,
    load_crew_benchmark_cases,
)


class TestLoadBenchmarkCases:
    def test_load_bare_array(self, tmp_path: Path):
        cases_data = [
            {"input": "test question", "expected": "answer", "criteria": "be helpful"}
        ]
        f = tmp_path / "cases.json"
        f.write_text(json.dumps(cases_data))

        cases, threshold = load_benchmark_cases(f)
        assert len(cases) == 1
        assert cases[0].input == "test question"
        assert cases[0].expected == "answer"
        assert threshold == 0.7

    def test_load_wrapper_object(self, tmp_path: Path):
        data = {
            "threshold": 0.9,
            "cases": [
                {"input": "q1", "expected": "a1"},
                {"input": "q2", "criteria": "be good"},
            ],
        }
        f = tmp_path / "cases.json"
        f.write_text(json.dumps(data))

        cases, threshold = load_benchmark_cases(f)
        assert len(cases) == 2
        assert threshold == 0.9

    def test_load_with_jsonc_comments(self, tmp_path: Path):
        jsonc = """[
  // Test case 1
  {"input": "hello", "expected": "hello"},
]"""
        f = tmp_path / "cases.jsonc"
        f.write_text(jsonc)

        cases, threshold = load_benchmark_cases(f)
        assert len(cases) == 1

    def test_load_invalid_format_raises(self, tmp_path: Path):
        f = tmp_path / "cases.json"
        f.write_text('"just a string"')

        with pytest.raises(ValueError, match="JSON array or an object"):
            load_benchmark_cases(f)

    def test_load_empty_cases(self, tmp_path: Path):
        f = tmp_path / "cases.json"
        f.write_text("[]")

        cases, _ = load_benchmark_cases(f)
        assert cases == []


class TestScoreCase:
    def test_no_expected_no_criteria_auto_passes(self):
        case = BenchmarkCase(input="test")
        passed, score = _score_case(case, "any response", "model")
        assert passed is True
        assert score == 1.0

    def test_expected_only_match(self):
        case = BenchmarkCase(input="test", expected="hello")
        passed, score = _score_case(case, "I said hello to you", "model")
        assert passed is True
        assert score == 1.0

    def test_expected_only_no_match(self):
        case = BenchmarkCase(input="test", expected="hello")
        passed, score = _score_case(case, "goodbye world", "model")
        assert passed is False
        assert score == 0.0

    def test_expected_case_insensitive(self):
        case = BenchmarkCase(input="test", expected="HELLO")
        passed, score = _score_case(case, "hello world", "model")
        assert passed is True
        assert score == 1.0


class TestParseJudgeResponse:
    def test_parse_json_response(self):
        raw = '{"score": 0.85, "passed": true, "reason": "good"}'
        passed, score = _parse_judge_response(raw)
        assert passed is True
        assert score == 0.85

    def test_parse_bare_number(self):
        raw = "The score is 0.9 based on quality."
        passed, score = _parse_judge_response(raw)
        assert passed is True
        assert score == 0.9

    def test_parse_failing_score(self):
        raw = '{"score": 0.3, "passed": false, "reason": "bad"}'
        passed, score = _parse_judge_response(raw)
        assert passed is False
        assert score == 0.3

    def test_parse_unparseable(self):
        raw = "I cannot score this."
        passed, score = _parse_judge_response(raw)
        assert passed is False
        assert score == 0.0

    def test_score_clamped_to_range(self):
        raw = '{"score": 1.5, "passed": true}'
        passed, score = _parse_judge_response(raw)
        assert score == 1.0

    def test_score_zero(self):
        raw = '{"score": 0.0, "passed": false}'
        passed, score = _parse_judge_response(raw)
        assert passed is False
        assert score == 0.0


class TestLoadCrewBenchmarkCases:
    def test_load_bare_array(self, tmp_path: Path):
        data = [
            {"inputs": {"topic": "AI"}, "criteria": "be insightful"}
        ]
        f = tmp_path / "benchmark.json"
        f.write_text(json.dumps(data))

        cases = load_crew_benchmark_cases(f)
        assert len(cases) == 1
        assert cases[0].inputs == {"topic": "AI"}
        assert cases[0].criteria == "be insightful"

    def test_load_wrapper_object(self, tmp_path: Path):
        data = {
            "cases": [
                {"inputs": {}, "criteria": "good output"},
                {"inputs": {"x": "1"}, "expected": "result"},
            ],
        }
        f = tmp_path / "benchmark.json"
        f.write_text(json.dumps(data))

        cases = load_crew_benchmark_cases(f)
        assert len(cases) == 2

    def test_defaults(self, tmp_path: Path):
        data = [{"criteria": "reasonable"}]
        f = tmp_path / "benchmark.json"
        f.write_text(json.dumps(data))

        cases = load_crew_benchmark_cases(f)
        assert cases[0].inputs == {}
        assert cases[0].expected is None

    def test_with_jsonc_comments(self, tmp_path: Path):
        jsonc = """[
  // Crew benchmark case
  {"inputs": {}, "criteria": "good"}
]"""
        f = tmp_path / "benchmark.jsonc"
        f.write_text(jsonc)

        cases = load_crew_benchmark_cases(f)
        assert len(cases) == 1


class TestCrewBenchmarkTimeout:
    def test_execute_crew_case_timeout_returns_timeout_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def slow_case(*_args, **_kwargs):
            time.sleep(1)
            return "done"

        monkeypatch.setattr(
            "crewai.project.benchmark._execute_crew_case_sync",
            slow_case,
        )

        result = _execute_crew_case_with_timeout(
            crew_path=tmp_path / "crew.jsonc",
            case=CrewBenchmarkCase(inputs={}),
            timeout=0.01,
        )

        assert result == "[TIMEOUT after 0.01s]"


class TestFindCheckpointBeforeAgent:
    def test_returns_none_when_no_checkpoint_dir(self, tmp_path: Path):
        crew_file = tmp_path / "crew.jsonc"
        crew_file.write_text(json.dumps({
            "agents": ["agent_a", "agent_b"],
            "tasks": [
                {"name": "t1", "description": "d", "expected_output": "e", "agent": "agent_a"},
                {"name": "t2", "description": "d", "expected_output": "e", "agent": "agent_b"},
            ],
        }))

        result = _find_checkpoint_before_agent(crew_file, "agent_b")
        assert result is None

    def test_returns_none_for_first_agent(self, tmp_path: Path):
        crew_file = tmp_path / "crew.jsonc"
        crew_file.write_text(json.dumps({
            "agents": ["agent_a"],
            "tasks": [
                {"name": "t1", "description": "d", "expected_output": "e", "agent": "agent_a"},
            ],
        }))

        result = _find_checkpoint_before_agent(crew_file, "agent_a")
        assert result is None

    def test_finds_valid_checkpoint(self, tmp_path: Path):
        crew_file = tmp_path / "crew.jsonc"
        crew_file.write_text(json.dumps({
            "agents": ["agent_a", "agent_b"],
            "tasks": [
                {"name": "t1", "description": "d", "expected_output": "e", "agent": "agent_a"},
                {"name": "t2", "description": "d", "expected_output": "e", "agent": "agent_b"},
            ],
        }))

        cp_dir = tmp_path / ".checkpoints" / "main"
        cp_dir.mkdir(parents=True)
        cp_file = cp_dir / "20260101T000000_abcd1234_p-none.json"
        cp_file.write_text(json.dumps({
            "entities": [{
                "tasks": [
                    {"output": {"raw": "task 1 done"}},
                    {"output": None},
                ],
            }],
        }))

        result = _find_checkpoint_before_agent(crew_file, "agent_b")
        assert result == cp_file

    def test_skips_invalid_checkpoint(self, tmp_path: Path):
        crew_file = tmp_path / "crew.jsonc"
        crew_file.write_text(json.dumps({
            "agents": ["agent_a", "agent_b"],
            "tasks": [
                {"name": "t1", "description": "d", "expected_output": "e", "agent": "agent_a"},
                {"name": "t2", "description": "d", "expected_output": "e", "agent": "agent_b"},
            ],
        }))

        cp_dir = tmp_path / ".checkpoints" / "main"
        cp_dir.mkdir(parents=True)
        # Both tasks complete — not useful for benchmarking agent_b
        cp_file = cp_dir / "20260101T000000_abcd1234_p-none.json"
        cp_file.write_text(json.dumps({
            "entities": [{
                "tasks": [
                    {"output": {"raw": "done"}},
                    {"output": {"raw": "also done"}},
                ],
            }],
        }))

        result = _find_checkpoint_before_agent(crew_file, "agent_b")
        assert result is None
