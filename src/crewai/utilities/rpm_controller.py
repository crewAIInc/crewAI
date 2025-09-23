import threading
import time
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from crewai.utilities.logger import Logger

"""Controls request rate limiting for API calls."""


class RPMController(BaseModel):
    """Manages requests per minute limiting."""

    max_rpm: Optional[int] = Field(default=None)
    logger: Logger = Field(default_factory=lambda: Logger(verbose=False))
    _current_rpm: int = PrivateAttr(default=0)
    _timer: Optional[threading.Timer] = PrivateAttr(default=None)
    _lock: Optional[threading.Lock] = PrivateAttr(default=None)
    _shutdown_flag: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def reset_counter(self):
        if self.max_rpm is not None:
            if not self._shutdown_flag:
                self._lock = threading.Lock()
                self._reset_request_count()
        return self

    def check_or_wait(self):
        if self.max_rpm is None:
            return True

        def _check_and_increment():
            if self.max_rpm is not None and self._current_rpm < self.max_rpm:
                self._current_rpm += 1
                return True
            elif self.max_rpm is not None:
                self.logger.log(
                    "info", "Max RPM reached, waiting for next minute to start."
                )
                self._wait_for_next_minute()
                self._current_rpm = 1
                return True
            return True

        if self._lock:
            with self._lock:
                return _check_and_increment()
        else:
            return _check_and_increment()

    def stop_rpm_counter(self):
        """Stop the RPM counter timer and mark for shutdown."""
        self._shutdown_flag = True
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        """Wait until the start of the next minute."""
        current_time = time.time()
        seconds_until_next_minute = 60 - (current_time % 60)
        time.sleep(seconds_until_next_minute)
        self._current_rpm = 0

    def _reset_request_count(self):
        """Reset the request count and schedule the next reset."""

        def _reset():
            self._current_rpm = 0
            if not self._shutdown_flag:
                self._timer = threading.Timer(60.0, self._reset_request_count)
                self._timer.daemon = True  # Ensure thread doesn't prevent program exit
                self._timer.start()

        if self._lock:
            with self._lock:
                _reset()
        else:
            _reset()

    def __del__(self):
        """Cleanup timer on object destruction."""
        try:
            if hasattr(self, "_timer") and self._timer:
                self._timer.cancel()
        except Exception:
            pass  # Ignore errors during cleanup
