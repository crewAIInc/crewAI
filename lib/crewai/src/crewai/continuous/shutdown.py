"""Shutdown controller for continuous operation mode."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from typing import Any


class ShutdownController:
    """Controller for managing graceful shutdown of continuous operations.

    This controller provides:
    - Graceful shutdown via request_stop()
    - Forced shutdown via force_stop()
    - Signal handling (SIGINT, SIGTERM)
    - Cleanup callback registration
    - Thread-safe shutdown coordination

    Example:
        controller = ShutdownController()

        # Register cleanup
        controller.on_cleanup(lambda: print("Cleaning up..."))

        # In your loop
        while not controller.should_stop:
            do_work()

        # Or stop externally
        controller.request_stop()
    """

    def __init__(self) -> None:
        """Initialize the shutdown controller."""
        self._stop_requested = threading.Event()
        self._force_stop = threading.Event()
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._original_sigint_handler: Any = None
        self._original_sigterm_handler: Any = None
        self._signals_installed = False

    @property
    def should_stop(self) -> bool:
        """Check if shutdown has been requested.

        Returns:
            True if stop has been requested (graceful or forced)
        """
        return self._stop_requested.is_set() or self._force_stop.is_set()

    @property
    def is_force_stop(self) -> bool:
        """Check if forced shutdown has been requested.

        Returns:
            True if forced stop has been requested
        """
        return self._force_stop.is_set()

    def request_stop(self) -> None:
        """Request a graceful shutdown.

        This sets the stop flag but allows current operations to complete.
        """
        self._stop_requested.set()

    def force_stop(self) -> None:
        """Request an immediate forced shutdown.

        This sets both stop flags to ensure immediate termination.
        """
        self._force_stop.set()
        self._stop_requested.set()

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        """Wait for a stop request.

        Args:
            timeout: Maximum time to wait in seconds. None for indefinite.

        Returns:
            True if stop was requested, False if timeout occurred
        """
        return self._stop_requested.wait(timeout=timeout)

    def on_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback to be called on shutdown.

        Args:
            callback: Function to call during cleanup
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def run_cleanup(self) -> None:
        """Run all registered cleanup callbacks.

        Callbacks are run in reverse order of registration (LIFO).
        Errors in callbacks are caught and logged but don't prevent
        other callbacks from running.
        """
        with self._lock:
            callbacks = list(reversed(self._cleanup_callbacks))

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                # Log but continue with other cleanups
                print(f"Error during cleanup: {e}")

    def install_signal_handlers(self) -> None:
        """Install signal handlers for SIGINT and SIGTERM.

        These handlers will request a graceful stop on first signal
        and force stop on second signal.
        """
        if self._signals_installed:
            return

        def signal_handler(signum: int, frame: Any) -> None:
            if self._stop_requested.is_set():
                # Second signal - force stop
                self.force_stop()
            else:
                # First signal - graceful stop
                self.request_stop()

        try:
            self._original_sigint_handler = signal.signal(
                signal.SIGINT, signal_handler
            )
            self._original_sigterm_handler = signal.signal(
                signal.SIGTERM, signal_handler
            )
            self._signals_installed = True
        except ValueError:
            # Signal handling not available (e.g., not main thread)
            pass

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if not self._signals_installed:
            return

        try:
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            if self._original_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            self._signals_installed = False
        except ValueError:
            pass

    def reset(self) -> None:
        """Reset the controller for reuse.

        Clears stop flags but keeps cleanup callbacks.
        """
        self._stop_requested.clear()
        self._force_stop.clear()

    def stop_with_timeout(self, timeout: float = 30.0) -> bool:
        """Request graceful stop and wait, then force if needed.

        Args:
            timeout: Time to wait for graceful stop before forcing

        Returns:
            True if stopped gracefully, False if forced
        """
        self.request_stop()

        # Wait for graceful stop
        start = time.time()
        while time.time() - start < timeout:
            if self._force_stop.is_set():
                return False
            time.sleep(0.1)

        # Timeout - force stop
        self.force_stop()
        return False

    def __enter__(self) -> "ShutdownController":
        """Context manager entry - install signal handlers."""
        self.install_signal_handlers()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - run cleanup and restore signals."""
        self.run_cleanup()
        self.restore_signal_handlers()
