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
    def setUp(self):
        self.tool = ZTDSSanitizerTool()

    def test_task_sanitization_and_restoration(self):
        session_id = "agent-task-01"
        mock_key = "".join(["ghp_", "1234567890abcdef", "1234567890abcdef"])
        raw_task = f"Deploy update for lead@partner.org with token {mock_key}"

        # 1. Sanitize before task execution
        sanitized = self.tool._run(raw_task, session_id=session_id)
        self.assertNotIn("lead@partner.org", sanitized)
        self.assertNotIn(mock_key, sanitized)
        self.assertIn("[EMAIL_TOKEN_1]", sanitized)
        self.assertIn("[API_SECRET_TOKEN_1]", sanitized)

        # 2. Restore after agent completion
        agent_output = "Completed deploy for [EMAIL_TOKEN_1] successfully."
        restored = self.tool.restore(agent_output, session_id=session_id)
        self.assertIn("lead@partner.org", restored)
        self.assertNotIn("[EMAIL_TOKEN_1]", restored)

        # 3. Invariant 3: Zeroization
        self.tool.zeroize(session_id=session_id)
        self.assertNotIn(session_id, self.tool._session_maps)
        self.assertNotIn(session_id, self.tool._entity_maps)

    def test_multitoken_ordering_safety(self):
        session_id = "agent-task-02"
        # First email in document must get TOKEN_1, tenth email must get TOKEN_10
        raw_task = " ".join([f"client{i}@corp.com" for i in range(1, 15)])
        sanitized = self.tool._run(raw_task, session_id=session_id)
        self.assertIn("[EMAIL_TOKEN_1]", sanitized)
        self.assertIn("[EMAIL_TOKEN_10]", sanitized)

        output = "Processed: [EMAIL_TOKEN_10] and [EMAIL_TOKEN_1]"
        restored = self.tool.restore(output, session_id=session_id)
        self.assertEqual(restored, "Processed: client10@corp.com and client1@corp.com")

    def test_extended_patterns(self):
        session_id = "agent-task-03"
        # Test long gTLD email, 15-digit Amex card, and hyphenated API secret
        mock_secret = "".join(["s", "k", "-proj-", "1234567890abcdef1234567890"])
        raw = f"Contact admin@cloud.technology or call with card 3782 822463 10005 using key {mock_secret}"
        sanitized = self.tool._run(raw, session_id=session_id)

        self.assertNotIn("admin@cloud.technology", sanitized)
        self.assertNotIn("3782 822463 10005", sanitized)
        self.assertNotIn(mock_secret, sanitized)
        self.assertIn("[EMAIL_TOKEN_1]", sanitized)
        self.assertIn("[CREDIT_CARD_TOKEN_1]", sanitized)
        self.assertIn("[API_SECRET_TOKEN_1]", sanitized)

        restored = self.tool.restore(sanitized, session_id=session_id)
        self.assertEqual(restored, raw)

    def test_token_collision_avoidance(self):
        session_id = "agent-task-collision"
        # Input text already contains literal [EMAIL_TOKEN_1]
        raw = "Contact alice@example.com but preserve [EMAIL_TOKEN_1] literal"
        sanitized = self.tool._run(raw, session_id=session_id)

        # alice@example.com must get [EMAIL_TOKEN_2] to avoid colliding with literal [EMAIL_TOKEN_1]
        self.assertIn("[EMAIL_TOKEN_2]", sanitized)
        self.assertIn("[EMAIL_TOKEN_1]", sanitized)
        self.assertEqual(sanitized, "Contact [EMAIL_TOKEN_2] but preserve [EMAIL_TOKEN_1] literal")

        # Restoring must only replace [EMAIL_TOKEN_2] back to alice@example.com, keeping [EMAIL_TOKEN_1]
        restored = self.tool.restore(sanitized, session_id=session_id)
        self.assertEqual(restored, raw)


if __name__ == "__main__":
    unittest.main()
