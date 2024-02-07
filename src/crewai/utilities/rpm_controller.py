import threading
import time
from typing import Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from crewai.utilities.logger import Logger


class RPMController(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    max_rpm: Union[int, None] = Field(default=None)
    logger: Logger = Field(default=None)
    _current_rpm: int = PrivateAttr(default=0)
    _timer: threading.Timer | None = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default=None)

    @model_validator(mode="after")
    def reset_counter(self):
        """        Reset the counter and acquire a lock if max_rpm is set.

        Returns:
            self

        Raises:
             Any exceptions that may occur during the execution of this method.
        """

        if self.max_rpm:
            self._lock = threading.Lock()
            self._reset_request_count()
        return self

    def check_or_wait(self):
        """        Check if the current RPM is less than the maximum RPM, and if so, increment the current RPM by 1.
        If the maximum RPM is reached, wait for the next minute to start and reset the current RPM to 1.

        Returns:
            True if the current RPM is less than the maximum RPM or after waiting for the next minute and resetting the current RPM.

        Raises:
             None
        """

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
        """        Stop the RPM counter.

        This method cancels the timer if it is running and sets the timer attribute to None.

        Returns:
            None
        """

        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        """        Wait for the next minute and reset the current RPM to 0.

        This method uses time.sleep to pause the execution for 60 seconds and then resets the current RPM to 0.

        Raises:
            Any exceptions raised by time.sleep.
        """

        time.sleep(60)
        self._current_rpm = 0

    def _reset_request_count(self):
        """        Reset the request count and start a new timer for resetting the count after 60 seconds.

        This method resets the request count to 0 and starts a new timer to reset the count after 60 seconds.

        Raises:
            <ExceptionType>: <Description of the exception raised>
        """

        with self._lock:
            self._current_rpm = 0
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(60.0, self._reset_request_count)
        self._timer.start()
