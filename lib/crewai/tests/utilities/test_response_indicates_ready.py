"""Tests for planning READY marker detection (#6204)."""

import pytest

from crewai.utilities.reasoning_handler import (
    AgentReasoning,
    response_indicates_ready,
)


class TestResponseIndicatesReady:
    """Unit tests for response_indicates_ready()."""

    def test_full_instructional_phrase(self) -> None:
        """完整就绪指令短语应判定为就绪。"""
        text = (
            "1. Gather data\n2. Analyze\n\n"
            "READY: I am ready to execute the task."
        )
        assert response_indicates_ready(text) is True

    def test_not_ready_overrides_full_instructional_phrase(self) -> None:
        """完整就绪短语后的 NOT READY 应作为最终判定。"""
        text = "READY: I am ready to execute the task.\nNOT READY"
        assert response_indicates_ready(text) is False

    def test_bare_ready_on_its_own_line(self) -> None:
        """独占一行的 READY 应判定为就绪。"""
        text = "Step 1: brainstorm ideas\nStep 2: refine them\n\nREADY\n"
        assert response_indicates_ready(text) is True

    def test_bare_ready_with_period(self) -> None:
        """带句号的独立 READY 标记应判定为就绪。"""
        assert response_indicates_ready("Plan looks solid.\n\nREADY.") is True

    def test_ready_with_colon_short_form(self) -> None:
        """带冒号的 READY 简写应判定为就绪。"""
        assert response_indicates_ready("Details...\nREADY: proceed") is True

    def test_not_ready_is_false(self) -> None:
        """独立的 NOT READY 标记应判定为未就绪。"""
        text = "Still missing context.\n\nNOT READY"
        assert response_indicates_ready(text) is False

    def test_not_ready_does_not_match_as_ready(self) -> None:
        """NOT READY 中的 READY 子串不应误判为就绪。"""
        assert response_indicates_ready("NOT READY") is False
        assert response_indicates_ready("NOT READY.") is False

    def test_mid_sentence_ready_is_not_a_marker(self) -> None:
        """句子中间的 ready 单词不应视为就绪标记。"""
        assert (
            response_indicates_ready("I am ready to begin researching sources.")
            is False
        )

    def test_last_marker_wins_after_refinement(self) -> None:
        """计划细化后应以最后一个 READY 标记为准。"""
        text = (
            "Initial thoughts...\nNOT READY\n"
            "Refined plan with missing pieces filled in.\nREADY"
        )
        assert response_indicates_ready(text) is True

    def test_last_marker_can_revert_to_not_ready(self) -> None:
        """最后一个 NOT READY 标记应撤销先前的就绪判定。"""
        text = "READY\nActually wait, still incomplete.\nNOT READY"
        assert response_indicates_ready(text) is False

    def test_empty_and_none_like(self) -> None:
        """空响应和仅含空白的响应应判定为未就绪。"""
        assert response_indicates_ready("") is False
        assert response_indicates_ready("   \n  ") is False

    def test_plan_without_marker_is_not_ready(self) -> None:
        """不含就绪标记的计划应判定为未就绪。"""
        assert response_indicates_ready("Just a plan with no conclusion.") is False

    def test_case_insensitive_ready_line(self) -> None:
        """就绪标记匹配应忽略字母大小写。"""
        assert response_indicates_ready("plan body\nready") is True
        assert response_indicates_ready("plan body\nnot ready") is False


class TestParsePlanningResponseReady:
    """Ensure _parse_planning_response uses the shared detector."""

    @pytest.mark.parametrize(
        ("response", "expected_ready"),
        [
            ("Plan...\nREADY: I am ready to execute the task.", True),
            ("Plan...\nREADY", True),
            ("Plan...\nNOT READY", False),
            ("", False),
        ],
    )
    def test_parse_planning_response(self, response: str, expected_ready: bool) -> None:
        """规划响应解析应复用共享的就绪检测逻辑。"""
        plan, ready = AgentReasoning._parse_planning_response(response)
        assert ready is expected_ready
        if response:
            assert plan == response
        else:
            assert plan == "No plan was generated."
