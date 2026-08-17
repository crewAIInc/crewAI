import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Direct import to allow standalone testing without full workspace virtualenv
_action_gate_file = (
    Path(__file__).resolve().parents[2] / "src" / "crewai" / "security" / "action_gate.py"
)
spec = importlib.util.spec_from_file_location("action_gate", str(_action_gate_file))
action_gate = importlib.util.module_from_spec(spec)
sys.modules["action_gate"] = action_gate
spec.loader.exec_module(action_gate)

ActionBoundary = action_gate.ActionBoundary
ActionGate = action_gate.ActionGate
ActionLedger = action_gate.ActionLedger
Disposition = action_gate.Disposition
ToolTier = action_gate.ToolTier


class TestActionGate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_file = Path(self.temp_dir) / "action_ledger.jsonl"
        self.ledger = ActionLedger(self.ledger_file)
        self.gate = ActionGate(ledger=self.ledger, prove_token=None)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if "AAG_KILL_SWITCH" in os.environ:
            del os.environ["AAG_KILL_SWITCH"]

    def test_read_tool_allowed(self):
        decision = self.gate.evaluate("search_customer_records", {"query": "Alice"})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.ALLOW)
        self.assertEqual(decision.tier, ToolTier.READ)
        self.assertFalse(decision.simulation_mode)
        self.assertTrue(len(decision.receipt_hash) == 64)

    def test_mutating_write_simulation_fallback(self):
        decision = self.gate.evaluate("create_user_account", {"username": "bob"})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.SIMULATE)
        self.assertEqual(decision.tier, ToolTier.WRITE_MUTATING)
        self.assertTrue(decision.simulation_mode)

    def test_mutating_write_authorized_with_token(self):
        gate_with_token = ActionGate(ledger=self.ledger, prove_token="valid-token-123")
        decision = gate_with_token.evaluate("create_user_account", {"username": "bob"})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.ALLOW)
        self.assertFalse(decision.simulation_mode)

    def test_destructive_tool_denied_without_token(self):
        decision = self.gate.evaluate("delete_database_table", {"table": "customers"})
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.DENY)
        self.assertEqual(decision.tier, ToolTier.DESTRUCTIVE)

    def test_destructive_tool_authorized_with_token(self):
        gate_with_token = ActionGate(ledger=self.ledger, prove_token="admin-token-999")
        decision = gate_with_token.evaluate("delete_database_table", {"table": "temp_staging"})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.ALLOW)

    def test_kill_switch_blocks_all(self):
        os.environ["AAG_KILL_SWITCH"] = "1"
        decision = self.gate.evaluate("get_profile", {"id": 1})
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.DENY)
        self.assertIn("Kill-switch engaged", decision.reason)

    def test_hash_chain_integrity(self):
        # Execute multiple actions
        self.gate.evaluate("get_user", {"id": 1})
        self.gate.evaluate("post_comment", {"text": "hello"})
        self.gate.evaluate("delete_cache", {"all": True})

        # Verify ledger file exists and contains 3 records
        lines = self.ledger_file.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 3)

        # Verify SHA-256 chain
        prev_hash = "0" * 64
        for line in lines:
            record = json.loads(line)
            self.assertEqual(record["prev_hash"], prev_hash)

            canonical_args = json.dumps(record["arguments"], sort_keys=True, default=str)
            expected_payload = (
                f"{prev_hash}|{record['timestamp']}|{record['tool_name']}|{canonical_args}|"
                f"{record['decision']['disposition']}|{record.get('agent_id') or ''}"
            )
            expected_hash = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
            self.assertEqual(record["receipt_hash"], expected_hash)
            prev_hash = record["receipt_hash"]

    def test_action_boundary_decorator_allow(self):
        boundary = ActionBoundary(self.gate)

        @boundary.guard
        def read_data(query: str):
            return f"Results for {query}"

        result = read_data(query="compliance")
        self.assertEqual(result, "Results for compliance")

    def test_action_boundary_decorator_simulate(self):
        boundary = ActionBoundary(self.gate)

        @boundary.guard
        def create_resource(name: str):
            return f"Resource {name} created"

        result = create_resource(name="prod-cluster")
        self.assertIsInstance(result, dict)
        self.assertTrue(result["simulation_mode"])
        self.assertEqual(result["disposition"], "simulate")

    def test_action_boundary_decorator_deny(self):
        boundary = ActionBoundary(self.gate)

        @boundary.guard
        def drop_database(db: str):
            return f"Database {db} dropped"

        with self.assertRaises(PermissionError) as ctx:
            drop_database(db="main")
        self.assertIn("ActionGate blocked tool", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
