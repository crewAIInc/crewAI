"""Controls request rate limiting for API calls."""

import threading

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.utilities.logger import Logger


class RPMController(BaseModel):
    """Manages requests per minute limiting."""

    max_rpm: int | None = Field(
        default=None,
        description="Maximum requests per minute. If None, no limit is applied.",
    )
    logger: Logger = Field(default_factory=lambda: Logger(verbose=False))
    _current_rpm: int = PrivateAttr(default=0)
    _timer: "threading.Timer | None" = PrivateAttr(default=None)
    _lock: "threading.Lock" = PrivateAttr(default_factory=threading.Lock)
    _shutdown_event: "threading.Event" = PrivateAttr(default_factory=threading.Event)

    @model_validator(mode="after")
    def reset_counter(self) -> Self:
        """Resets the RPM counter and starts the timer if max_rpm is set.

        Returns:
            The instance of the RPMController.
        """
        if self.max_rpm is not None:
            if not self._shutdown_event.is_set():
                self._reset_request_count()
        return self

    def check_or_wait(self) -> bool:
        """Checks if a new request can be made based on the RPM limit.

        If the limit is reached, waits (up to 60 seconds) for the counter to
        reset. The wait is performed **outside** the lock so that other threads
        are not blocked.

        Returns:
            True if a new request can be made, False otherwise.
        """
        if self.max_rpm is None:
            return True

        with self._lock:
            if self._current_rpm < self.max_rpm:
                self._current_rpm += 1
                return True

        # Limit reached — wait outside the lock so other threads aren't blocked.
        self.logger.log(
            "info", "Max RPM reached, waiting for next minute to start."
        )
        self._shutdown_event.wait(60)

        with self._lock:
            self._current_rpm = 1
        return True

    def stop_rpm_counter(self) -> None:
        """Stops the RPM counter and cancels any active timers."""
        self._shutdown_event.set()
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

    def _reset_request_count(self) -> None:
        """Resets the current RPM count and schedules the next reset.

        The lock is held during the entire operation to prevent races between
        the timer callback, ``check_or_wait``, and ``stop_rpm_counter``.
        """
        with self._lock:
            self._current_rpm = 0
            if not self._shutdown_event.is_set():
                self._timer = threading.Timer(60.0, self._reset_request_count)
                self._timer.daemon = True
                self._timer.start()
