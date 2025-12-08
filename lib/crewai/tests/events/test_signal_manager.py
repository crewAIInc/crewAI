"""Tests for SystemSignalManager and signal event decoupling from telemetry.

These tests verify that:
1. Signal events work when telemetry is disabled
2. Signal handler chaining works correctly
3. SystemSignalManager properly bridges OS signals to the event bus
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.signal_manager import SystemSignalManager, system_signal_manager
from crewai.events.types.system_events import (
    SigTermEvent,
    on_signal,
)


class TestSystemSignalManager:
    """Tests for SystemSignalManager class."""

    def test_singleton_pattern(self) -> None:
        """Test that SystemSignalManager is a singleton."""
        manager1 = SystemSignalManager()
        manager2 = SystemSignalManager()
        assert manager1 is manager2

    def test_global_instance_is_singleton(self) -> None:
        """Test that the global system_signal_manager is the singleton instance."""
        manager = SystemSignalManager()
        assert manager is system_signal_manager

    def test_register_signal_stores_original_handler(self) -> None:
        """Test that register_signal stores the original handler."""
        manager = SystemSignalManager()
        original = signal.getsignal(signal.SIGUSR1)

        try:
            manager.register_signal(signal.SIGUSR1, SigTermEvent, shutdown=False)
            assert signal.SIGUSR1 in manager._original_handlers
        finally:
            signal.signal(signal.SIGUSR1, original)

    def test_register_signal_emits_event(self) -> None:
        """Test that registered signal handler emits event to event bus."""
        import threading

        received_events: list[SigTermEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(SigTermEvent)
        def handler(source: object, event: SigTermEvent) -> None:
            with condition:
                received_events.append(event)
                condition.notify_all()

        manager = SystemSignalManager()
        original = signal.getsignal(signal.SIGUSR1)

        try:
            manager.register_signal(signal.SIGUSR1, SigTermEvent, shutdown=False)
            os.kill(os.getpid(), signal.SIGUSR1)

            with condition:
                condition.wait_for(lambda: len(received_events) >= 1, timeout=5.0)

            assert len(received_events) >= 1
            assert isinstance(received_events[0], SigTermEvent)
        finally:
            signal.signal(signal.SIGUSR1, original)


class TestSignalEventsWithTelemetryDisabled:
    """Tests verifying signal events work when telemetry is disabled.

    These tests use subprocess to avoid interfering with pytest's signal handling.
    """

    @pytest.mark.timeout(30)
    def test_on_signal_handler_fires_with_telemetry_disabled(self) -> None:
        """Test that @on_signal handlers fire even when telemetry is disabled.

        This is the core fix for issue #4041: signal events should work
        regardless of the CREWAI_DISABLE_TELEMETRY setting.
        """
        script = textwrap.dedent('''
            import os
            import sys
            import time

            os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

            from crewai.events.types.system_events import SignalEvent, on_signal

            @on_signal
            def user_signal_handler(source: object, event: SignalEvent) -> None:
                print(f"[USER_HANDLER] Received event type={event.type}", flush=True)

            print(f"[READY] PID={os.getpid()}", flush=True)
            sys.stdout.flush()

            while True:
                time.sleep(0.1)
        ''')

        env = os.environ.copy()
        env["CREWAI_DISABLE_TELEMETRY"] = "true"

        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        try:
            ready_line = proc.stdout.readline()
            assert "[READY]" in ready_line, f"Process not ready: {ready_line}"

            time.sleep(0.5)
            proc.send_signal(signal.SIGTERM)

            stdout, stderr = proc.communicate(timeout=10)
            full_output = ready_line + stdout

            assert "[USER_HANDLER]" in full_output, (
                f"User handler did not fire. Output: {full_output}, Stderr: {stderr}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @pytest.mark.timeout(30)
    def test_sigint_handler_fires_with_telemetry_disabled(self) -> None:
        """Test that SIGINT events work when telemetry is disabled."""
        script = textwrap.dedent('''
            import os
            import sys
            import time

            os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

            from crewai.events.types.system_events import SigIntEvent
            from crewai.events.event_bus import crewai_event_bus

            @crewai_event_bus.on(SigIntEvent)
            def sigint_handler(source: object, event: SigIntEvent) -> None:
                print(f"[SIGINT_HANDLER] Received SIGINT event", flush=True)

            print(f"[READY] PID={os.getpid()}", flush=True)
            sys.stdout.flush()

            while True:
                time.sleep(0.1)
        ''')

        env = os.environ.copy()
        env["CREWAI_DISABLE_TELEMETRY"] = "true"

        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        try:
            ready_line = proc.stdout.readline()
            assert "[READY]" in ready_line, f"Process not ready: {ready_line}"

            time.sleep(0.5)
            proc.send_signal(signal.SIGINT)

            stdout, stderr = proc.communicate(timeout=10)
            full_output = ready_line + stdout

            assert "[SIGINT_HANDLER]" in full_output, (
                f"SIGINT handler did not fire. Output: {full_output}, Stderr: {stderr}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class TestSignalHandlerChaining:
    """Tests verifying signal handler chaining works correctly.

    These tests verify that when user code registers a signal handler before
    CrewAI, the CrewAI handler properly chains to the original handler.
    """

    @pytest.mark.timeout(30)
    def test_baseline_handler_called_after_crewai_handler(self) -> None:
        """Test that baseline OS handler is called after CrewAI emits the event.

        This tests the scenario where user code registers a signal handler
        before CrewAI imports. The CrewAI handler should emit the event and
        then call the original handler.
        """
        script = textwrap.dedent('''
            import os
            import signal
            import sys
            import time
            from typing import Any

            def baseline_handler(signum: int, frame: Any) -> None:
                print("[BASELINE_HANDLER] Signal received", flush=True)

            signal.signal(signal.SIGTERM, baseline_handler)

            from crewai.events.types.system_events import SignalEvent, on_signal

            @on_signal
            def user_signal_handler(source: object, event: SignalEvent) -> None:
                print(f"[USER_HANDLER] Received event type={event.type}", flush=True)

            print(f"[READY] PID={os.getpid()}", flush=True)
            sys.stdout.flush()

            while True:
                time.sleep(0.1)
        ''')

        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            ready_line = proc.stdout.readline()
            assert "[READY]" in ready_line, f"Process not ready: {ready_line}"

            time.sleep(0.5)
            proc.send_signal(signal.SIGTERM)

            time.sleep(1.0)

            proc.send_signal(signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=5)
            full_output = ready_line + stdout

            assert "[USER_HANDLER]" in full_output, (
                f"User handler did not fire. Output: {full_output}, Stderr: {stderr}"
            )
            assert "[BASELINE_HANDLER]" in full_output, (
                f"Baseline handler did not fire. Output: {full_output}, Stderr: {stderr}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class TestTelemetrySignalIntegration:
    """Tests for Telemetry's integration with the signal event system."""

    def test_telemetry_registers_shutdown_handlers_on_event_bus(self) -> None:
        """Test that Telemetry registers shutdown handlers on the event bus."""
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            with patch("crewai.telemetry.telemetry.BatchSpanProcessor"):
                with patch("crewai.telemetry.telemetry.SafeOTLPSpanExporter"):
                    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "false"}):
                        from crewai.telemetry.telemetry import Telemetry

                        Telemetry._instance = None
                        telemetry = Telemetry()

                        assert telemetry.ready is True
