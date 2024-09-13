import threading
import time
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from crewai.utilities.logger import Logger


class RPMController(BaseModel):
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
        print("check_or_waitcheck_or_waitcheck_or_waitcheck_or_wait")
        if self.max_rpm is None:
            return True

        def _check_and_increment():
            print(
                "_check_and_increment_check_and_increment_check_and_increment_check_and_increment"
            )
            if self.max_rpm is not None and self._current_rpm < self.max_rpm:
                self._current_rpm += 1
                print("111111111111111")
                print("self._current_rpm", self._current_rpm)
                print("self.max_rpm", self.max_rpm)
                return True
            elif self.max_rpm is not None:
                print("22222222222222")
                self.logger.log(
                    "info", "Max RPM reached, waiting for next minute to start."
                )
                print("CARALHO")
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
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        time.sleep(1)
        self._current_rpm = 0

    def _reset_request_count(self):
        def _reset():
            self._current_rpm = 0
            if not self._shutdown_flag:
                self._timer = threading.Timer(60.0, self._reset_request_count)
                self._timer.start()

        if self._lock:
            with self._lock:
                _reset()
        else:
            _reset()

        if self._timer:
            self._shutdown_flag = True
            self._timer.cancel()
