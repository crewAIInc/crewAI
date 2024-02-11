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
    _shutdown_flag = False

    @model_validator(mode="after")
    def reset_counter(self):
        if self.max_rpm:
            if not self._shutdown_flag:
                self._lock = threading.Lock()
                self._reset_request_count()
        return self

    def check_or_wait(self):
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
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        time.sleep(60)
        self._current_rpm = 0

    def _reset_request_count(self):
        with self._lock:
            self._current_rpm = 0
        if self._timer:
            self._shutdown_flag = True
            self._timer.cancel()
        self._timer = threading.Timer(60.0, self._reset_request_count)
        self._timer.start()
