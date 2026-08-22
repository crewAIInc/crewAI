import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
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
LedgerIntegrityError = action_gate.LedgerIntegrityError


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


class TestActionGateSecurityFixes(unittest.TestCase):
    """Regression tests for a security review that found the token check,
    tool classification, ledger tamper-evidence, concurrency safety, and
    argument logging were all exploitable in the original implementation.
    Each test below reproduces the original exploit and asserts it is
    now blocked (or, for redaction, that the leak no longer occurs).
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_file = Path(self.temp_dir) / "action_ledger.jsonl"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _fresh_ledger_view(self, path):
        """A read-only ActionLedger-shaped view for calling verify_chain()
        without triggering __init__'s recovery (which raises on a
        tampered file -- exactly what some of these tests are inducing)."""
        inst = ActionLedger.__new__(ActionLedger)
        inst.ledger_path = path
        inst._thread_lock = threading.Lock()
        return inst

    # ---- Token check: a per-call token must be VERIFIED, not merely truthy

    def test_arbitrary_caller_token_no_longer_authorizes_destructive_op(self):
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token="real-secret-only-operator-knows")
        decision = gate.evaluate(
            "delete_production_database", {}, prove_token="literally-anything-works"
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.DENY)

    def test_correct_explicit_token_still_authorizes(self):
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token="real-secret-only-operator-knows")
        decision = gate.evaluate(
            "delete_production_database", {}, prove_token="real-secret-only-operator-knows"
        )
        self.assertTrue(decision.allowed)

    def test_gates_own_configured_token_still_authorizes_without_override(self):
        """The pre-existing, intended behavior: a gate deployed with its own
        secret authorizes calls that don't pass a per-call override."""
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token="ops-deployed-secret")
        decision = gate.evaluate("delete_staging_cache", {})
        self.assertTrue(decision.allowed)

    def test_no_secret_configured_anywhere_means_nothing_can_be_proven(self):
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token=None)
        decision = gate.evaluate("delete_x", {}, prove_token="anything")
        self.assertFalse(decision.allowed)

    # ---- Tool classification: severity-ordered, whole-token matching

    def test_destructive_tool_with_read_like_name_is_not_misclassified(self):
        """Original bug: dict-insertion-order substring matching classified
        this as READ (the 'search'/'get' pattern matched before 'terminate'
        was reached), which bypassed ALL gating -- READ tools execute
        unconditionally with no token check at all."""
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token=None)
        decision = gate.evaluate("search_and_terminate_instance", {})
        self.assertEqual(decision.tier, ToolTier.DESTRUCTIVE)
        self.assertFalse(decision.allowed)

    def test_whole_token_matching_avoids_false_positive_substrings(self):
        """'budget_report' contains the raw substring 'get' (inside 'bud-
        GET') but is not a get_* tool. The original substring-'in' check
        would have matched it as READ; whole-token matching must not."""
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token=None)
        self.assertIn("get", "budget_report")  # confirms the substring IS present
        decision = gate.evaluate("budget_report", {})
        self.assertNotEqual(decision.tier, ToolTier.READ)
        # No real token matches either -> falls through to the safe default.
        self.assertEqual(decision.tier, ToolTier.WRITE_MUTATING)

    # ---- Ledger tamper-evidence: verify_chain() must actually verify

    def test_verify_chain_passes_on_an_untampered_ledger(self):
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token=None)
        gate.evaluate("get_user", {"id": 1})
        gate.evaluate("post_comment", {"text": "hello"})
        ok, reason = ledger.verify_chain()
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_verify_chain_detects_a_tampered_entry(self):
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token=None)
        gate.evaluate("get_user", {"id": 1})
        gate.evaluate("post_comment", {"text": "hello"})

        lines = self.ledger_file.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[0])
        entry["arguments"]["id"] = 999  # tamper with a past, already-hashed entry
        lines[0] = json.dumps(entry)
        self.ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ok, reason = self._fresh_ledger_view(self.ledger_file).verify_chain()
        self.assertFalse(ok)
        self.assertIn("receipt_hash", reason)

    def test_recovery_refuses_to_continue_on_a_tampered_ledger(self):
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token=None)
        gate.evaluate("get_user", {"id": 1})

        lines = self.ledger_file.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[0])
        entry["prev_hash"] = "f" * 64  # tamper with the chain linkage itself
        lines[0] = json.dumps(entry)
        self.ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with self.assertRaises(LedgerIntegrityError):
            ActionLedger(self.ledger_file)

    # ---- Sensitive argument redaction

    def test_sensitive_arguments_are_redacted_in_the_ledger_but_not_in_the_call(self):
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token="secret")
        boundary = ActionBoundary(gate)
        received = {}

        @boundary.guard
        def get_account(username: str, password: str):
            received["password"] = password  # the real function must see the real value
            return "ok"

        get_account(username="alice", password="hunter2-the-real-secret")

        self.assertEqual(received["password"], "hunter2-the-real-secret")
        logged = json.loads(self.ledger_file.read_text(encoding="utf-8").strip())
        self.assertEqual(logged["arguments"]["password"], "***REDACTED***")
        self.assertEqual(logged["arguments"]["username"], "alice")

    # ---- Concurrency safety

    def test_concurrent_writes_from_multiple_threads_do_not_fork_the_chain(self):
        ledger = ActionLedger(self.ledger_file)
        gate = ActionGate(ledger=ledger, prove_token=None)

        def worker(n):
            for i in range(20):
                gate.evaluate(f"get_item_{n}_{i}", {"n": n, "i": i})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = self.ledger_file.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 160)
        ok, reason = self._fresh_ledger_view(self.ledger_file).verify_chain()
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
