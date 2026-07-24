import pytest

from crewai.utilities.reasoning_handler import _is_ready

class TestReasoningHandlerIsReady:
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("READY", True),
            ("ready", True),
            ("READY: I am ready to execute the task.", True),
            ("I am READY to go.", True),
            ("Status: ready", True),
            ("NOT READY", False),
            ("not ready", False),
            ("NOT  READY", False),
            ("NOT\nREADY", False),
            ("NOT\tREADY", False),
            ("NOT READY: I need to refine.", False),
            ("The agent is NOT READY to proceed.", False),
            ("I am NOT    READY.", False),
            ("NOT ready at all", False),
            ("Something else completely", False),
            ("", False),
        ],
    )
    def test_is_ready_detects_ready_and_not_ready_correctly(self, response, expected):
        assert _is_ready(response) == expected
