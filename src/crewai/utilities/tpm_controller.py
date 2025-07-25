import threading
import time
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict

from crewai.utilities.logger import Logger
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess

"""Controls Token rate ."""


class TPMController(BaseModel):
    """Manages Tokens per minute limiting."""

    max_tpm: Optional[int] = Field(default=None)
    logger: Logger = Field(default_factory=lambda: Logger(verbose=False))
    token_counter: TokenProcess 
    _current_tokens: int = PrivateAttr(default=0)
    _timer: Optional[threading.Timer] = PrivateAttr(default=None)
    _lock: Optional[threading.Lock] = PrivateAttr(default=None)
    _shutdown_flag: bool = PrivateAttr(default=False)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def reset_counter(self):
        if self.max_tpm is not None:
            if not self._shutdown_flag:
                self._lock = threading.Lock()
                self._reset_request_count()
        return self

    def check_or_wait(self, wait: int = 0):
        if self.max_tpm is None:
            return True

        def _check_and_increment(wait):
          
            if self.max_tpm is not None and self._current_tokens < self.max_tpm and not wait:
                print("Tokens checked")
                self._current_tokens += self.token_counter.total_tokens
                print(f"Tokens increased: {self._current_tokens}")

                return True
            elif self.max_tpm is not None:
                self.logger.log(
                    "info", "Max TPM reached, waiting for next minute to start."
                )
                self._wait_for_next_minute()
                
                return True
            return True

        if self._lock:
            with self._lock:
                return _check_and_increment(wait)
        else:
            return _check_and_increment(wait)

    def stop_tpm_counter(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        time.sleep(60)
        self._current_tokens = 0
    
    def external_wait_for_next_minute(self):
        if self._lock:
            with self._lock:
                pass
        else:            
            time.sleep(60)
            self._current_tokens = 0        

    def _reset_request_count(self):
        def _reset():
            self._current_tokens = 0
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
