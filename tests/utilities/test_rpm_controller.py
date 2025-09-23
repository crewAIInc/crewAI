"""Tests for RPMController rate limiting functionality."""

import threading
from unittest.mock import Mock, patch
import pytest

from crewai.utilities.rpm_controller import RPMController


class TestRPMController:
    """Test suite for RPMController."""

    def test_no_limit_when_max_rpm_is_none(self):
        """Test that no rate limiting is applied when max_rpm is None."""
        controller = RPMController(max_rpm=None)

        # Should always return True when no limit is set
        for _ in range(100):
            assert controller.check_or_wait() is True

        # No timer should be created
        assert controller._timer is None

    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        controller = RPMController(max_rpm=5)

        # First 5 requests should pass immediately
        for i in range(5):
            assert controller.check_or_wait() is True
            assert controller._current_rpm == i + 1

        # 6th request should trigger waiting
        with patch.object(controller, "_wait_for_next_minute") as mock_wait:
            assert controller.check_or_wait() is True
            mock_wait.assert_called_once()
            # After waiting, counter should be reset to 1
            assert controller._current_rpm == 1

    def test_timer_initialization(self):
        """Test that timer is properly initialized."""
        controller = RPMController(max_rpm=10)

        # Timer should be created
        assert controller._timer is not None
        assert isinstance(controller._timer, threading.Timer)
        assert controller._timer.daemon is True

        # Cleanup
        controller.stop_rpm_counter()

    def test_timer_cleanup_on_stop(self):
        """Test that timer is properly cleaned up when stopped."""
        controller = RPMController(max_rpm=10)

        # Timer should be created
        assert controller._timer is not None

        # Stop the counter
        controller.stop_rpm_counter()

        # Timer should be cleaned up
        assert controller._timer is None
        assert controller._shutdown_flag is True

    def test_timer_cleanup_on_deletion(self):
        """Test that timer is cleaned up when object is deleted."""
        controller = RPMController(max_rpm=10)
        timer = controller._timer

        # Delete the controller
        del controller

        # Timer should be cancelled (we can't directly check this,
        # but we can verify it doesn't throw an exception)
        assert timer is not None

    def test_wait_for_next_minute_calculation(self):
        """Test that wait time is calculated correctly."""
        controller = RPMController(max_rpm=5)

        with patch("time.time") as mock_time, patch("time.sleep") as mock_sleep:
            # Simulate being 45 seconds into the current minute
            mock_time.return_value = 45.0

            controller._wait_for_next_minute()

            # Should wait 15 seconds to reach the next minute
            mock_sleep.assert_called_once_with(15.0)
            assert controller._current_rpm == 0

    def test_wait_for_next_minute_edge_cases(self):
        """Test wait time calculation for edge cases."""
        controller = RPMController(max_rpm=5)

        test_cases = [
            (0.0, 60.0),  # Start of minute
            (30.0, 30.0),  # Middle of minute
            (59.0, 1.0),  # End of minute
            (59.9, 0.1),  # Very end of minute
        ]

        for current_time, expected_wait in test_cases:
            with patch("time.time") as mock_time, patch("time.sleep") as mock_sleep:
                mock_time.return_value = current_time

                controller._wait_for_next_minute()

                # Allow small floating point differences
                actual_wait = mock_sleep.call_args[0][0]
                assert abs(actual_wait - expected_wait) < 0.01

    def test_thread_safety(self):
        """Test that the controller is thread-safe."""
        controller = RPMController(max_rpm=100)
        results = []

        def make_requests(num_requests):
            local_results = []
            for _ in range(num_requests):
                result = controller.check_or_wait()
                local_results.append(result)
            results.extend(local_results)

        # Create multiple threads making concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_requests, args=(10,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should have succeeded
        assert all(results)
        assert len(results) == 100

        # Cleanup
        controller.stop_rpm_counter()

    def test_counter_reset_after_minute(self):
        """Test that counter resets after a minute."""
        controller = RPMController(max_rpm=5)

        # Make 5 requests
        for _ in range(5):
            controller.check_or_wait()

        assert controller._current_rpm == 5

        # Manually trigger reset (simulating timer callback)
        controller._reset_request_count()

        # Counter should be reset
        assert controller._current_rpm == 0

        # Should be able to make more requests
        for i in range(5):
            assert controller.check_or_wait() is True
            assert controller._current_rpm == i + 1

        # Cleanup
        controller.stop_rpm_counter()

    def test_no_timer_after_shutdown(self):
        """Test that no new timer is created after shutdown."""
        controller = RPMController(max_rpm=10)

        # Stop the controller
        controller.stop_rpm_counter()

        # Try to reset counter - should not create new timer
        controller._reset_request_count()

        # Timer should still be None
        assert controller._timer is None
        assert controller._shutdown_flag is True

    def test_logger_output(self):
        """Test that logger is called when rate limit is reached."""
        from crewai.utilities.logger import Logger

        mock_logger = Mock(spec=Logger)
        controller = RPMController(max_rpm=2, logger=mock_logger)

        # First 2 requests should not log
        controller.check_or_wait()
        controller.check_or_wait()
        mock_logger.log.assert_not_called()

        # 3rd request should log
        with patch.object(controller, "_wait_for_next_minute"):
            controller.check_or_wait()
            mock_logger.log.assert_called_once_with(
                "info", "Max RPM reached, waiting for next minute to start."
            )

        # Cleanup
        controller.stop_rpm_counter()

    def test_multiple_instances(self):
        """Test that multiple instances work independently."""
        controller1 = RPMController(max_rpm=3)
        controller2 = RPMController(max_rpm=5)

        # Each controller should track its own count
        for _ in range(3):
            controller1.check_or_wait()
        for _ in range(5):
            controller2.check_or_wait()

        assert controller1._current_rpm == 3
        assert controller2._current_rpm == 5

        # Cleanup
        controller1.stop_rpm_counter()
        controller2.stop_rpm_counter()

    @pytest.mark.parametrize("max_rpm", [1, 10, 100, 1000])
    def test_various_limits(self, max_rpm):
        """Test with various rate limits."""
        controller = RPMController(max_rpm=max_rpm)

        # Should allow exactly max_rpm requests
        for i in range(max_rpm):
            assert controller.check_or_wait() is True
            assert controller._current_rpm == i + 1

        # Next request should trigger waiting
        with patch.object(controller, "_wait_for_next_minute"):
            controller.check_or_wait()
            assert controller._current_rpm == 1

        # Cleanup
        controller.stop_rpm_counter()
