"""System signal manager for CrewAI.

This module provides a singleton manager that bridges OS signals to the CrewAI
event bus, independent of telemetry settings. This ensures that signal events
(SigTermEvent, SigIntEvent, etc.) are always emitted when signals are received,
regardless of whether telemetry is enabled or disabled.
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from crewai.events.event_bus import crewai_event_bus


if TYPE_CHECKING:
    from collections.abc import Callable

    from crewai.events.base_events import BaseEvent

    EventFactory = Callable[[], BaseEvent]

logger = logging.getLogger(__name__)


class SystemSignalManager:
    """Singleton manager for bridging OS signals to the CrewAI event bus.

    This class registers signal handlers that emit corresponding events to the
    event bus, allowing any code to listen for system signals via the event
    system. It operates independently of telemetry settings.

    The manager supports handler chaining: when a signal handler is registered,
    it preserves any previously registered handler and calls it after emitting
    the event. This allows user code to register handlers before or after
    CrewAI initialization.

    Attributes:
        _instance: Singleton instance of the manager.
        _lock: Thread lock for singleton initialization.
        _original_handlers: Mapping of signals to their original handlers.
        _registered_signals: Set of signals that have been registered.
    """

    _instance: Self | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> Self:
        """Create or return the singleton instance.

        Returns:
            The singleton SystemSignalManager instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the signal manager.

        This is safe to call multiple times; initialization only happens once.
        """
        if getattr(self, "_initialized", False):
            return

        self._initialized: bool = True
        self._original_handlers: dict[signal.Signals, Any] = {}
        self._registered_signals: set[signal.Signals] = set()
        self._handler_lock = threading.Lock()

    def register_signal(
        self,
        sig: signal.Signals,
        event_factory: EventFactory,
        shutdown: bool = False,
    ) -> None:
        """Register a signal handler that emits an event to the event bus.

        This method can be called multiple times for the same signal. Each call
        will re-read the current signal handler and wrap it, ensuring that any
        handlers registered after the initial setup are still called.

        Args:
            sig: The signal to handle (e.g., signal.SIGTERM).
            event_factory: A callable that creates the event to emit.
            shutdown: If True, raise SystemExit(0) after handling if there
                     was no original handler to call.
        """
        with self._handler_lock:
            try:
                original_handler = signal.getsignal(sig)
                self._original_handlers[sig] = original_handler

                def handler(signum: int, frame: Any) -> None:
                    crewai_event_bus.emit(self, event_factory())

                    if original_handler not in (signal.SIG_DFL, signal.SIG_IGN, None):
                        if callable(original_handler):
                            original_handler(signum, frame)
                    elif shutdown:
                        raise SystemExit(0)

                signal.signal(sig, handler)
                self._registered_signals.add(sig)
            except ValueError as e:
                logger.warning(
                    f"Cannot register {sig.name} handler: not running in main thread",
                    exc_info=e,
                )
            except OSError as e:
                logger.warning(f"Cannot register {sig.name} handler: {e}", exc_info=e)

    def ensure_handlers_installed(self) -> None:
        """Ensure signal handlers are installed, re-wrapping if necessary.

        This method can be called to reinstall signal handlers, which is useful
        when user code has registered handlers after CrewAI's initial setup.
        The handlers will be re-registered to wrap any new handlers that were
        installed since the last registration.

        This is a no-op if called before any signals have been registered via
        register_signal().
        """


system_signal_manager: SystemSignalManager = SystemSignalManager()
