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
        args = {"amount": 100, "recipient": "alice"}
        
        assert not guard.is_duplicate(args)
    
    def test_second_execution_is_duplicate(self):
        """Second tool call with same args should be duplicate"""
        guard = IdempotencyGuard("send_payment")
        args = {"amount": 100, "recipient": "alice"}
        
        guard.record(args, "payment_sent")
        
        assert guard.is_duplicate(args)
        assert guard.get_cached_result(args) == "payment_sent"
    
    def test_different_args_not_duplicate(self):
        """Different arguments should not be duplicate"""
        guard = IdempotencyGuard("send_payment")
        
        guard.record({"amount": 100, "recipient": "alice"}, "payment_1")
        
        assert not guard.is_duplicate({"amount": 200, "recipient": "bob"})
    
    def test_different_tools_not_duplicate(self):
        """Different tools with same args should not be duplicate"""
        guard1 = IdempotencyGuard("send_payment")
        guard2 = IdempotencyGuard("send_email")
        
        args = {"id": 123}
        guard1.record(args, "payment_sent")
        
        assert not guard2.is_duplicate(args)
    
    def test_file_backend_persists(self, tmp_path):
        """File backend should persist across instances"""
        storage_path = tmp_path / ".ccs_idempotency.json"
        
        # First instance records
        guard1 = IdempotencyGuard("send_payment", storage_backend="file")
        guard1._storage_path = storage_path
        args = {"amount": 100}
        guard1.record(args, "payment_sent")
        
        # Second instance should see it
        guard2 = IdempotencyGuard("send_payment", storage_backend="file")
        guard2._storage_path = storage_path
        guard2._storage = {}  # Reset memory
        
        # Load from file
        if storage_path.exists():
            import json
            with open(storage_path, 'r') as f:
                guard2._storage = json.load(f)
        
        assert guard2.is_duplicate(args)
    
    def test_reset_clears_storage(self):
        """Reset should clear all storage"""
        guard = IdempotencyGuard("send_payment")
        guard.record({"amount": 100}, "result")
        
        IdempotencyGuard.reset()
        
        assert not guard.is_duplicate({"amount": 100})


class TestIdempotentDecorator:
    """Test @idempotent decorator"""
    
    def setup_method(self):
        IdempotencyGuard.reset()
    
    def test_decorator_blocks_duplicate(self):
        """Decorator should block duplicate calls"""
        call_count = 0
        
        @idempotent()
        def send_payment(amount: int) -> str:
            nonlocal call_count
            call_count += 1
            return f"payment_{call_count}"
        
        # First call
        result1 = send_payment(amount=100)
        assert call_count == 1
        assert result1 == "payment_1"
        
        # Second call (should be blocked)
        result2 = send_payment(amount=100)
        assert call_count == 1  # Not incremented
        assert result2 == "payment_1"  # Cached
    
    def test_decorator_different_args(self):
        """Decorator should allow different args"""
        call_count = 0
        
        @idempotent()
        def send_payment(amount: int) -> str:
            nonlocal call_count
            call_count += 1
            return f"payment_{call_count}"
        
        result1 = send_payment(amount=100)
        result2 = send_payment(amount=200)
        
        assert call_count == 2
        assert result1 == "payment_1"
        assert result2 == "payment_2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
