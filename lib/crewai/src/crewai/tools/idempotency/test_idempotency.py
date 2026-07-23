"""
Tests for CrewAI #5802 fix: Idempotency guard for tool retries
"""
import pytest
from pathlib import Path
from idempotency import IdempotencyGuard, idempotent


class TestIdempotencyGuard:
    """Test IdempotencyGuard core functionality"""

    def setup_method(self):
        IdempotencyGuard.reset()

    def test_first_execution_not_duplicate(self):
        """First tool call should not be detected as duplicate"""
        guard = IdempotencyGuard("send_payment")
        call_key = {"args": (100, "alice"), "kwargs": {}}

        assert not guard.is_duplicate(call_key)

    def test_second_execution_is_duplicate(self):
        """Second tool call with same args should be duplicate"""
        guard = IdempotencyGuard("send_payment")
        call_key = {"args": (), "kwargs": {"amount": 100, "recipient": "alice"}}

        guard.record(call_key, "payment_sent")

        assert guard.is_duplicate(call_key)
        assert guard.get_cached_result(call_key) == "payment_sent"

    def test_different_args_not_duplicate(self):
        """Different arguments should not be duplicate"""
        guard = IdempotencyGuard("send_payment")

        guard.record({"args": (), "kwargs": {"amount": 100, "recipient": "alice"}}, "payment_1")

        assert not guard.is_duplicate({"args": (), "kwargs": {"amount": 200, "recipient": "bob"}})

    def test_positional_args_distinguished(self):
        """Positional args must produce different keys (fix for CodeRabbit Critical)"""
        guard = IdempotencyGuard("send_payment")

        key1 = {"args": (100, "alice"), "kwargs": {}}
        key2 = {"args": (200, "bob"), "kwargs": {}}

        guard.record(key1, "payment_1")

        assert guard.is_duplicate(key1)
        assert not guard.is_duplicate(key2)

    def test_mixed_args_and_kwargs(self):
        """Mix of positional and keyword args should be handled correctly"""
        guard = IdempotencyGuard("send_payment")

        key1 = {"args": (100,), "kwargs": {"recipient": "alice"}}
        key2 = {"args": (200,), "kwargs": {"recipient": "bob"}}
        key3 = {"args": (100,), "kwargs": {"recipient": "alice"}}

        guard.record(key1, "payment_1")

        assert guard.is_duplicate(key3)  # Same args -> duplicate
        assert not guard.is_duplicate(key2)  # Different args -> not duplicate

    def test_different_tools_not_duplicate(self):
        """Different tools with same args should not be duplicate"""
        guard1 = IdempotencyGuard("send_payment")
        guard2 = IdempotencyGuard("send_email")

        call_key = {"args": (123,), "kwargs": {}}
        guard1.record(call_key, "payment_sent")

        assert not guard2.is_duplicate(call_key)

    def test_file_backend_persists(self, tmp_path):
        """File backend should persist across instances"""
        storage_path = tmp_path / ".ccs_idempotency.json"

        # First instance records
        guard1 = IdempotencyGuard("send_payment", storage_backend="file")
        guard1._storage_path = storage_path
        call_key = {"args": (100, "alice"), "kwargs": {}}
        guard1.record(call_key, "payment_sent")

        # Second instance should see the record
        guard2 = IdempotencyGuard("send_payment", storage_backend="file")
        guard2._storage_path = storage_path
        guard2._storage = {}  # simulate fresh load
        import json
        if storage_path.exists():
            with open(storage_path, "r") as f:
                guard2._storage = json.load(f)

        assert guard2.is_duplicate(call_key)
        assert guard2.get_cached_result(call_key) == "payment_sent"


class TestIdempotentDecorator:
    """Test @idempotent decorator"""

    def setup_method(self):
        IdempotencyGuard.reset()

    def test_decorator_blocks_duplicate(self):
        """Decorator should block duplicate calls with same args"""
        call_count = 0

        @idempotent()
        def send_payment(amount, recipient):
            nonlocal call_count
            call_count += 1
            return f"sent {amount} to {recipient}"

        # First call executes
        r1 = send_payment(100, "alice")
        assert r1 == "sent 100 to alice"
        assert call_count == 1

        # Second call with same args returns cached
        r2 = send_payment(100, "alice")
        assert r2 == "sent 100 to alice"
        assert call_count == 1  # Not executed again

    def test_decorator_allows_different_args(self):
        """Decorator should allow calls with different args"""
        call_count = 0

        @idempotent()
        def send_payment(amount, recipient):
            nonlocal call_count
            call_count += 1
            return f"sent {amount} to {recipient}"

        r1 = send_payment(100, "alice")
        r2 = send_payment(200, "bob")

        assert r1 == "sent 100 to alice"
        assert r2 == "sent 200 to bob"
        assert call_count == 2  # Both executed

    def test_decorator_with_kwargs_only(self):
        """Decorator should work with keyword-only calls"""
        call_count = 0

        @idempotent()
        def send_payment(amount=0, recipient=""):
            nonlocal call_count
            call_count += 1
            return f"sent {amount} to {recipient}"

        r1 = send_payment(amount=100, recipient="alice")
        r2 = send_payment(amount=100, recipient="alice")

        assert call_count == 1  # Second call blocked
        assert r2 == r1

    def test_decorator_positional_vs_kwargs(self):
        """Decorator should distinguish positional from keyword calls"""
        call_count = 0

        @idempotent()
        def send_payment(amount, recipient):
            nonlocal call_count
            call_count += 1
            return f"sent {amount} to {recipient}"

        # Positional
        send_payment(100, "alice")
        assert call_count == 1

        # Same values as kwargs — should still be different call key
        send_payment(amount=100, recipient="alice")
        assert call_count == 2  # Different call key format
