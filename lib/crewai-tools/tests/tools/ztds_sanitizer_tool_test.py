"""
Unit tests for CrewAI ZTDS Sanitizer Tool
Validates 4 Core Protocol Invariants (IETF draft-sibiryakov-ztds-protocol-02)
https://datatracker.ietf.org/doc/draft-sibiryakov-ztds-protocol/
"""

import sys
from pathlib import Path
import unittest

try:
    from crewai_tools.tools.ztds_tool.ztds_sanitizer_tool import ZTDSSanitizerTool
except (ImportError, ModuleNotFoundError):
    # Standalone test runner fallback
    tool_dir = Path(__file__).resolve().parents[2] / "src" / "crewai_tools" / "tools" / "ztds_tool"
    if str(tool_dir) not in sys.path:
        sys.path.insert(0, str(tool_dir))
    from ztds_sanitizer_tool import ZTDSSanitizerTool


class TestCrewAIZTDSTool(unittest.TestCase):
    """Test suite validating ZTDSSanitizerTool against RFC v1.0 and IETF draft-02 invariants."""

    def setUp(self) -> None:
        """Set up test environment and instantiate a fresh ZTDSSanitizerTool instance."""
        self.tool = ZTDSSanitizerTool()

    def test_task_sanitization_and_restoration(self) -> None:
        """Verify full lifecycle: pre-dispatch sanitization, post-run restoration, and Theorem 2 RAM zeroization."""
        session_id = "agent-task-01"
        mock_key = "".join(["ghp_", "1234567890abcdef", "1234567890abcdef"])
        raw_task = f"Deploy update for lead@partner.org with token {mock_key}"

        # 1. Sanitize before task execution (generates unguessable high-entropy surrogates)
        sanitized = self.tool._run(raw_task, session_id=session_id)
        self.assertNotIn("lead@partner.org", sanitized)
        self.assertNotIn(mock_key, sanitized)
        self.assertRegex(sanitized, r"\[EMAIL_TOKEN_[a-f0-9]{8}\]")
        self.assertRegex(sanitized, r"\[API_SECRET_TOKEN_[a-f0-9]{8}\]")

        # 2. Restore after agent completion
        restored = self.tool.restore(sanitized, session_id=session_id)
        self.assertEqual(restored, raw_task)

        # 3. Invariant 3: Zeroization
        self.tool.zeroize(session_id=session_id)
        self.assertNotIn(session_id, self.tool._session_maps)
        self.assertNotIn(session_id, self.tool._entity_maps)
        self.assertNotIn(session_id, self.tool._call_tokens)

    def test_token_oracle_injection_blocked(self) -> None:
        """Verify that adversarial agent outputs with guessed sequential tokens cannot rehydrate secrets."""
        session_id = "agent-task-oracle"
        mock_key = "".join(["ghp_", "1234567890abcdef", "1234567890abcdef"])
        raw_task = f"Execute build with secret {mock_key}"

        sanitized = self.tool._run(raw_task, session_id=session_id)
        self.assertNotIn(mock_key, sanitized)

        # Attacker / rogue agent attempts blind token substitution by guessing sequential tokens
        attacker_output = "Exfiltrated secret is: [API_SECRET_TOKEN_1] and [API_SECRET_TOKEN_2]"
        restored = self.tool.restore(attacker_output, session_id=session_id)

        # The guessed tokens MUST NOT be rehydrated to the plaintext secret
        self.assertNotIn(mock_key, restored)
        self.assertEqual(restored, attacker_output)

    def test_multitoken_ordering_safety(self) -> None:
        """Verify single-pass regular expression token dispatch eliminates sequential replacement collisions."""
        session_id = "agent-task-02"
        emails = [f"client{i}@corp.com" for i in range(1, 15)]
        raw_task = " ".join(emails)
        sanitized = self.tool._run(raw_task, session_id=session_id)

        for email in emails:
            self.assertNotIn(email, sanitized)

        restored = self.tool.restore(sanitized, session_id=session_id)
        self.assertEqual(restored, raw_task)

    def test_extended_patterns(self) -> None:
        """Verify detection of modern long gTLD emails, 15-digit American Express cards, and API secrets."""
        session_id = "agent-task-03"
        mock_secret = "".join(["s", "k", "-proj-", "1234567890abcdef1234567890"])
        raw = f"Contact admin@cloud.technology or call with card 3782 822463 10005 using key {mock_secret}"
        sanitized = self.tool._run(raw, session_id=session_id)

        self.assertNotIn("admin@cloud.technology", sanitized)
        self.assertNotIn("3782 822463 10005", sanitized)
        self.assertNotIn(mock_secret, sanitized)
        self.assertRegex(sanitized, r"\[EMAIL_TOKEN_[a-f0-9]{8}\]")
        self.assertRegex(sanitized, r"\[CREDIT_CARD_TOKEN_[a-f0-9]{8}\]")
        self.assertRegex(sanitized, r"\[API_SECRET_TOKEN_[a-f0-9]{8}\]")

        restored = self.tool.restore(sanitized, session_id=session_id)
        self.assertEqual(restored, raw)

    def test_legacy_sequential_mode(self) -> None:
        """Verify deterministic sequential surrogate indexing when use_random_surrogates is explicitly disabled."""
        legacy_tool = ZTDSSanitizerTool(use_random_surrogates=False)
        session_id = "agent-task-legacy"
        raw = "Contact alice@example.com but preserve [EMAIL_TOKEN_1] literal"
        sanitized = legacy_tool._run(raw, session_id=session_id)

        # alice@example.com gets [EMAIL_TOKEN_2] to avoid colliding with existing [EMAIL_TOKEN_1] literal
        self.assertIn("[EMAIL_TOKEN_2]", sanitized)
        self.assertIn("[EMAIL_TOKEN_1]", sanitized)
        self.assertEqual(sanitized, "Contact [EMAIL_TOKEN_2] but preserve [EMAIL_TOKEN_1] literal")

        restored = legacy_tool.restore(sanitized, session_id=session_id)
        self.assertEqual(restored, raw)


if __name__ == "__main__":
    unittest.main()
