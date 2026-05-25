"""Tests for crewai.project.benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crewai.project.benchmark import (
    BenchmarkCase,
    _parse_judge_response,
    _score_case,
    load_benchmark_cases,
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
