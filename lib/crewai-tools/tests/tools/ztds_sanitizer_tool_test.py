"""
Unit tests for CrewAI ZTDS Sanitizer Tool
Validates 4 Core Protocol Invariants (IETF draft-sibiryakov-ztds-protocol-02)
https://datatracker.ietf.org/doc/draft-sibiryakov-ztds-protocol/
"""

import unittest
from crewai_tools.tools.ztds_tool.ztds_sanitizer_tool import ZTDSSanitizerTool


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
        agent_output = f"Completed deploy for [EMAIL_TOKEN_1] successfully."
        restored = self.tool.restore(agent_output, session_id=session_id)
        self.assertIn("lead@partner.org", restored)
        self.assertNotIn("[EMAIL_TOKEN_1]", restored)

        # 3. Invariant 3: Zeroization
        self.tool.zeroize(session_id=session_id)
        self.assertNotIn(session_id, self.tool._session_maps)
        self.assertNotIn(session_id, self.tool._entity_maps)


    def test_multitoken_ordering_safety(self):
        session_id = "agent-task-02"
        raw_task = " ".join([f"client{i}@corp.com" for i in range(1, 15)])
        self.tool._run(raw_task, session_id=session_id)
        output = "Processed: [EMAIL_TOKEN_10] and [EMAIL_TOKEN_1]"
        restored = self.tool.restore(output, session_id=session_id)
        self.assertIn("client10@corp.com", restored)
        self.assertIn("client1@corp.com", restored)

if __name__ == "__main__":
    unittest.main()
