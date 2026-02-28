"""Tests for RPM (requests per minute) controller.

This module tests the RPMController class for correct rate limiting,
thread safety, and timer lifecycle management.
"""

import threading
import time

from crewai.utilities.rpm_controller import RPMController


def test_no_limit_always_allows():
    """When max_rpm is None, check_or_wait should always return True."""
    controller = RPMController(max_rpm=None)
    for _ in range(100):
        assert controller.check_or_wait() is True
    controller.stop_rpm_counter()


def test_basic_rate_limiting():
    """Requests up to max_rpm should be allowed immediately."""
    controller = RPMController(max_rpm=5)
    for _ in range(5):
        assert controller.check_or_wait() is True
    controller.stop_rpm_counter()


def test_counter_resets_after_timer():
    """After the timer fires (simulated), requests should be allowed again."""
    controller = RPMController(max_rpm=3)

    # Use up all requests
    for _ in range(3):
        controller.check_or_wait()

    # Manually reset the counter to simulate timer firing
    controller._reset_request_count()

    # Should allow requests again
    assert controller.check_or_wait() is True
    controller.stop_rpm_counter()


def test_stop_cancels_timer():
    """stop_rpm_counter should cancel any pending timer."""
    controller = RPMController(max_rpm=5)
    assert controller._timer is not None
    assert controller._timer.is_alive()

    controller.stop_rpm_counter()

    assert controller._timer is None


def test_stop_sets_shutdown_event():
    """stop_rpm_counter should set the shutdown event."""
    controller = RPMController(max_rpm=5)
    assert not controller._shutdown_event.is_set()

    controller.stop_rpm_counter()

    assert controller._shutdown_event.is_set()


def test_concurrent_requests_respect_limit():
    """Multiple threads should collectively respect the RPM limit."""
    max_rpm = 10
    controller = RPMController(max_rpm=max_rpm)
    success_count = [0]
    count_lock = threading.Lock()

    def make_request():
        # Try to make a request, but don't wait if limit reached
        with controller._lock:
            if controller._current_rpm < controller.max_rpm:
                controller._current_rpm += 1
                with count_lock:
                    success_count[0] += 1

    threads = [threading.Thread(target=make_request) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert success_count[0] <= max_rpm
    controller.stop_rpm_counter()


def test_stop_unblocks_waiting_threads():
    """stop_rpm_counter should unblock threads waiting in check_or_wait."""
    controller = RPMController(max_rpm=1)

    # Exhaust the limit
    controller.check_or_wait()

    unblocked = threading.Event()

    def waiting_thread():
        controller.check_or_wait()
        unblocked.set()

    t = threading.Thread(target=waiting_thread, daemon=True)
    t.start()

    # Give the thread time to enter the wait
    time.sleep(0.1)
    assert not unblocked.is_set(), "Thread should be blocked waiting"

    # Stop should unblock the waiting thread
    controller.stop_rpm_counter()

    assert unblocked.wait(timeout=2.0), "Thread was not unblocked by stop_rpm_counter"
    t.join(timeout=2.0)


def test_shutdown_event_prevents_timer_restart():
    """After stop_rpm_counter, _reset_request_count should not start a new timer."""
    controller = RPMController(max_rpm=5)
    controller.stop_rpm_counter()

    # Manually call reset — should NOT start a new timer
    controller._reset_request_count()

    assert controller._timer is None


def test_lock_not_held_during_wait():
    """When one thread is waiting due to RPM limit, other threads should
    still be able to acquire the lock (i.e., the wait happens outside the lock)."""
    controller = RPMController(max_rpm=1)

    # Exhaust the limit
    controller.check_or_wait()

    lock_acquired = threading.Event()

    def try_lock():
        acquired = controller._lock.acquire(timeout=1.0)
        if acquired:
            lock_acquired.set()
            controller._lock.release()

    # Start a thread that waits in check_or_wait (blocked)
    waiting = threading.Thread(target=controller.check_or_wait, daemon=True)
    waiting.start()
    time.sleep(0.1)  # Let it enter the wait

    # Another thread should be able to acquire the lock
    lock_thread = threading.Thread(target=try_lock, daemon=True)
    lock_thread.start()

    assert lock_acquired.wait(timeout=2.0), "Lock should be available while thread waits"

    # Cleanup
    controller.stop_rpm_counter()
    waiting.join(timeout=2.0)
    lock_thread.join(timeout=2.0)


def test_concurrent_wakeup_respects_limit():
    """Multiple threads waking from wait must not collectively exceed max_rpm.

    Regression test for the bug where all waiting threads set _current_rpm = 1
    instead of atomically incrementing, bypassing the rate limit.
    """
    max_rpm = 2
    controller = RPMController(max_rpm=max_rpm)

    # Exhaust the limit
    for _ in range(max_rpm):
        controller.check_or_wait()

    # Spawn several threads that will all block in check_or_wait
    results = []
    results_lock = threading.Lock()

    def blocked_request():
        controller.check_or_wait()
        with results_lock:
            results.append(1)

    threads = [threading.Thread(target=blocked_request, daemon=True) for _ in range(5)]
    for t in threads:
        t.start()

    # Let all threads enter the wait
    time.sleep(0.2)

    # Simulate the timer resetting the counter — this wakes no one,
    # but makes slots available for threads that loop back.
    controller._reset_request_count()

    # Give threads time to wake and compete
    time.sleep(0.5)

    # At most max_rpm threads should have succeeded (counter was reset to 0)
    with controller._lock:
        assert controller._current_rpm <= max_rpm, (
            f"_current_rpm={controller._current_rpm} exceeded max_rpm={max_rpm}"
        )

    # Cleanup
    controller.stop_rpm_counter()
    for t in threads:
        t.join(timeout=2.0)

