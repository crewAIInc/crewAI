import threading
import time

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from crewai.utilities.logger import Logger

"""Controls request rate limiting for API calls."""


class RPMController(BaseModel):
    """Manages requests per minute limiting."""

    max_rpm: int | None = Field(default=None)
    logger: Logger = Field(default_factory=lambda: Logger(verbose=False))
    _current_rpm: int = PrivateAttr(default=0)
    _timer: threading.Timer | None = PrivateAttr(default=None)
    _lock: threading.Lock | None = PrivateAttr(default=None)
    _shutdown_flag: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def reset_counter(self):
        if self.max_rpm is not None and not self._shutdown_flag:
            self._lock = threading.Lock()
            self._reset_request_count()
        return self

    def check_or_wait(self):
        if self.max_rpm is None:
            return True

        def _check_and_increment() -> bool:
            if self.max_rpm is not None and self._current_rpm < self.max_rpm:
                self._current_rpm += 1
                return True
            if self.max_rpm is not None:
                self.logger.log(
                    "info", "Max RPM reached, waiting for next minute to start.",
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

    def stop_rpm_counter(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self) -> None:
        time.sleep(60)
        self._current_rpm = 0

    def _reset_request_count(self) -> None:
        def _reset() -> None:
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
