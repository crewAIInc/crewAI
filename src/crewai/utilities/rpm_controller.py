import threading
import time
from typing import Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from crewai.utilities.logger import Logger


class RPMController(BaseModel):
    """This class is responsible for controlling the rate of requests per minute (RPM).
    It uses a threading.Timer to reset the request count every minute."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Configuration dictionary for the model
    max_rpm: Union[int, None] = Field(default=None)  # Maximum allowed requests per minute
    logger: Logger = Field(default=None)  # Logger instance for logging messages
    _current_rpm: int = PrivateAttr(default=0)  # Current number of requests in this minute
    _timer: threading.Timer | None = PrivateAttr(default=None)  # Timer for resetting the request count
    _lock: threading.Lock = PrivateAttr(default=None)  # Lock for thread-safe operations
    _shutdown_flag = False  # Flag to indicate if the controller is shutting down

    @model_validator(mode="after")
    def reset_counter(self):
        """Resets the request counter if the max_rpm is set and the controller is not shutting down.
        It initializes the lock and resets the request count."""
        if self.max_rpm:
            if not self._shutdown_flag:
                self._lock = threading.Lock()
                self._reset_request_count()
        return self

    def check_or_wait(self):
        """Checks if a new request can be made or waits until the next minute if the max_rpm is reached.
        It increments the current_rpm and returns True if a new request can be made.
        If the max_rpm is reached, it logs a message, waits for the next minute, resets the current_rpm, and returns True."""
        if not self.max_rpm:
            return True

        with self._lock:
            if self._current_rpm < self.max_rpm:
                self._current_rpm += 1
                return True
            else:
                self.logger.log(
                    "info", "Max RPM reached, waiting for next minute to start."
                )
                self._wait_for_next_minute()
                self._current_rpm = 1
                return True

    def stop_rpm_counter(self):
        """Stops the RPM counter by cancelling the timer."""
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        """Waits for the next minute by sleeping for 60 seconds and then resets the current_rpm."""
        time.sleep(60)
        self._current_rpm = 0

    def _reset_request_count(self):
        """Resets the request count to 0 in a thread-safe manner.
        If a timer is already running, it cancels the timer and starts a new one to reset the request count after 60 seconds."""
        with self._lock:
            self._current_rpm = 0
        if self._timer:
            self._shutdown_flag = True
            self._timer.cancel()
        self._timer = threading.Timer(60.0, self._reset_request_count)
        self._timer.start()
