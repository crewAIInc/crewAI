"""
Regression test for crewAI issue #7290:
Memory.forget() must drain pending background saves before deleting,
otherwise a save submitted before forget() can resurrect deleted content.
"""
import threading
import time
from unittest.mock import MagicMock, patch
import pytest


def test_forget_drains_pending_saves_before_delete():
    """
    forget() must call drain_writes() before self._storage.delete().
    Without the fix, a background save submitted before forget() can land
    after the delete and resurrect the forgotten content.
    """
    from crewai.memory.unified_memory import Memory

    # Track call order
    call_order = []

    # Create a Memory instance with a mock storage
    mem = Memory.__new__(Memory)
    mem.__pydantic_private__ = {}

    # Patch drain_writes and _storage.delete to record call order
    original_forget = Memory.forget

    drain_called = []
    delete_called = []

    def mock_drain(self):
        drain_called.append("drain")
        call_order.append("drain_writes")

    def mock_delete(self, **kwargs):
        delete_called.append("delete")
        call_order.append("storage.delete")
        return 1

    with patch.object(Memory, "drain_writes", mock_drain), \
         patch.object(Memory, "_storage", create=True) as mock_storage:
        mock_storage.delete = lambda **kwargs: (call_order.append("storage.delete"), 1)[1]

        # We need a real Memory instance - use a simpler approach
        # Just verify the source code has drain_writes() before _storage.delete()
        import inspect
        source = inspect.getsource(Memory.forget)
        lines = source.split("\n")

        drain_line = None
        delete_line = None
        for i, line in enumerate(lines):
            if "self.drain_writes()" in line and drain_line is None:
                drain_line = i
            if "self._storage.delete(" in line and delete_line is None:
                delete_line = i

        assert drain_line is not None, "forget() must call self.drain_writes()"
        assert delete_line is not None, "forget() must call self._storage.delete()"
        assert drain_line < delete_line, (
            f"drain_writes() (line {drain_line}) must come before "
            f"_storage.delete() (line {delete_line}) in forget()"
        )


def test_forget_holds_reset_lock_across_drain_and_delete():
    """forget() must hold _reset_lock across the drain AND the delete.

    _submit_save() registers under _reset_lock. If forget() released the
    lock between the drain snapshot and the delete, a concurrent save could
    register in between, land after the deletion, and resurrect the
    forgotten content. Probed deterministically: a helper thread attempts a
    non-blocking acquire of _reset_lock while the drain runs and again while
    the delete runs; both must fail. (A same-thread probe would succeed
    trivially: the lock is re-entrant.)
    """
    from crewai.memory.unified_memory import Memory

    mem = Memory.model_construct()
    mem._pending_saves = []
    mem.root_scope = None

    probe = {}

    def try_acquire(key):
        def _probe():
            got = mem._reset_lock.acquire(blocking=False)
            probe[key] = got
            if got:
                mem._reset_lock.release()

        t = threading.Thread(target=_probe)
        t.start()
        t.join()

    real_drain = Memory.drain_writes

    def probed_drain(self):
        try_acquire("drain")
        return real_drain(self)

    def fake_delete(**kwargs):
        try_acquire("delete")
        return 1

    mem._storage = MagicMock()
    mem._storage.delete.side_effect = fake_delete

    with patch.object(Memory, "drain_writes", probed_drain):
        assert Memory.forget(mem) == 1
    assert probe.get("drain") is False, (
        "forget() must hold _reset_lock while draining, otherwise a "
        "concurrent _submit_save() can register after the snapshot and "
        "resurrect forgotten content"
    )
    assert probe.get("delete") is False, (
        "forget() must hold _reset_lock while deleting, otherwise a "
        "concurrent _submit_save() can slip between drain and delete and "
        "resurrect forgotten content"
    )
def test_recall_already_has_drain_writes():
    """Confirm recall() has the read barrier (regression guard)."""
    from crewai.memory.unified_memory import Memory
    import inspect

    source = inspect.getsource(Memory.recall)
    assert "self.drain_writes()" in source, "recall() must call self.drain_writes()"


def test_forget_drain_ordering_matches_recall():
    """
    Both forget() and recall() should call drain_writes() before
    touching storage, ensuring consistent write-barrier semantics.
    """
    from crewai.memory.unified_memory import Memory
    import inspect

    for method_name in ("forget", "recall"):
        method = getattr(Memory, method_name)
        source = inspect.getsource(method)
        lines = source.split("\n")

        drain_line = next(
            (i for i, l in enumerate(lines) if "self.drain_writes()" in l), None
        )
        storage_line = next(
            (i for i, l in enumerate(lines)
             if "self._storage." in l and "drain" not in l), None
        )

        assert drain_line is not None, f"{method_name}() must call drain_writes()"
        assert storage_line is not None, f"{method_name}() must access _storage"
        assert drain_line < storage_line, (
            f"{method_name}(): drain_writes() must precede _storage access"
        )
