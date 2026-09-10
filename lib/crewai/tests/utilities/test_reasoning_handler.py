import unittest

from crewai.utilities.reasoning_handler import AgentReasoning


class TestPlanReadyDetection(unittest.TestCase):
    """Regression tests for AgentReasoning._is_plan_ready (issue #6204).

    The original implementation matched the exact sentence
    "READY: I am ready to execute the task." so a bare "READY" from the model
    was never detected. The regex replacement must keep detecting a standalone
    READY while still rejecting NOT READY.
    """

    def test_bare_ready_is_detected(self) -> None:
        # The exact shape reported in issue #6204.
        self.assertTrue(AgentReasoning._is_plan_ready("READY"))

    def test_canonical_ready_sentence_is_detected(self) -> None:
        self.assertTrue(
            AgentReasoning._is_plan_ready("READY: I am ready to execute the task.")
        )

    def test_ready_detection_is_case_insensitive(self) -> None:
        for variant in ("Ready", "ready", "ReAdY"):
            with self.subTest(variant=variant):
                self.assertTrue(AgentReasoning._is_plan_ready(variant))

    def test_not_ready_is_rejected(self) -> None:
        for variant in ("NOT READY", "not ready", "Not Ready"):
            with self.subTest(variant=variant):
                self.assertFalse(AgentReasoning._is_plan_ready(variant))

    def test_not_ready_with_extra_whitespace_is_rejected(self) -> None:
        # A fixed-width lookbehind only excluded one whitespace character, so
        # these were wrongly reported as ready.
        for variant in ("NOT  READY", "NOT   READY", "NOT\tREADY", "NOT\nREADY"):
            with self.subTest(variant=variant):
                self.assertFalse(AgentReasoning._is_plan_ready(variant))

    def test_substrings_containing_ready_are_rejected(self) -> None:
        for variant in ("already done", "readiness check", "unready"):
            with self.subTest(variant=variant):
                self.assertFalse(AgentReasoning._is_plan_ready(variant))

    def test_standalone_ready_later_in_the_text_is_detected(self) -> None:
        self.assertTrue(
            AgentReasoning._is_plan_ready("I am not ready, but the plan is ready")
        )


if __name__ == "__main__":
    unittest.main()
