"""The terminal UI resolves local-session consent without the legacy uploader."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from crewai_cli.crew_run_tui import CrewRunApp, TraceConsentScreen
import pytest
from textual.events import Mount
from textual.widgets import Button


class ConsentApp(CrewRunApp):
    def on_mount(self, event: Mount | None = None) -> None:
        """Show the real UI without starting a crew or a refresh worker."""
        if event is not None:
            event.prevent_default()


async def wait_for_consent(app, pilot):
    for _ in range(50):
        await pilot.pause(0.01)
        if isinstance(app.screen, TraceConsentScreen):
            return app.screen
    raise AssertionError("The execution worker did not open the consent prompt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "approved"),
    [
        ("y", True),
        ("n", False),
        ("escape", False),
        ("click-yes", True),
        ("click-no", False),
    ],
)
async def test_session_consent_buttons_and_keys_resolve_execution_worker(
    monkeypatch, action, approved
):
    app = ConsentApp()
    legacy = Mock(side_effect=AssertionError("Session consent invoked legacy upload"))
    monkeypatch.setattr(app, "_send_traces_worker", legacy, raising=False)
    async with app.run_test(size=(100, 40)) as pilot:
        result = asyncio.create_task(asyncio.to_thread(app._request_trace_consent))
        screen = await wait_for_consent(app, pilot)
        assert str(screen.query_one("#btn-consent-yes", Button).label) == "Share Trace"
        content = screen._build_content().plain
        assert "stored locally" in content and "Sharing uploads it" in content
        assert "prompts, inputs, outputs, and tool calls" in content
        assert "20 seconds" in content
        assert not result.done()
        if action.startswith("click-"):
            assert await pilot.click(f"#btn-consent-{action.removeprefix('click-')}")
        else:
            await pilot.press(action)
        assert await asyncio.wait_for(result, timeout=5) is approved
        await pilot.pause()
        assert not isinstance(app.screen, TraceConsentScreen)
    legacy.assert_not_called()
    assert app._trace_consent_pending is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["true", "1"])
async def test_tracing_asked_for_needs_no_modal(monkeypatch, value):
    """The user turned tracing on for this project; the modal would be the same
    question a second time."""
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", value)
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.utils._is_interactive_terminal", lambda: True
    )
    app = ConsentApp()
    async with app.run_test(size=(100, 40)) as pilot:
        assert await asyncio.to_thread(app._request_trace_consent) is True
        await pilot.pause()
        assert not isinstance(app.screen, TraceConsentScreen)
    assert app._trace_consent_pending is None


@pytest.mark.asyncio
async def test_a_declaration_is_the_same_yes_as_the_variable(monkeypatch):
    """`tracing=True` never reaches this worker's context — the crew set it on
    the thread it was built on — so the declaration itself is read."""
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    app = ConsentApp()
    app._crew = SimpleNamespace(tracing=True)
    async with app.run_test(size=(100, 40)) as pilot:
        assert await asyncio.to_thread(app._request_trace_consent) is True
        await pilot.pause()
        assert not isinstance(app.screen, TraceConsentScreen)


@pytest.mark.asyncio
async def test_consent_timeout_returns_false_and_dismisses_modal(monkeypatch):
    app = ConsentApp()
    waits = []
    event = threading.Event()

    class ExpiredEvent:
        def set(self):
            event.set()

        def wait(self, *, timeout):
            waits.append(timeout)
            return False

    # Replace only this module's event factory, leaving Textual's threads alone.
    monkeypatch.setattr(
        "crewai_cli.crew_run_tui.threading", SimpleNamespace(Event=ExpiredEvent)
    )
    legacy = Mock(side_effect=AssertionError("Timeout invoked legacy upload"))
    monkeypatch.setattr(app, "_send_traces_worker", legacy, raising=False)
    async with app.run_test(size=(100, 40)) as pilot:
        result = await asyncio.wait_for(
            asyncio.to_thread(app._request_trace_consent), timeout=5
        )
        assert result is False
        await pilot.pause()
        assert not isinstance(app.screen, TraceConsentScreen)
    assert waits == [20]
    assert app._trace_consent_pending is None
    legacy.assert_not_called()


@pytest.mark.asyncio
async def test_quit_rejects_pending_consent_and_releases_worker(monkeypatch):
    app = ConsentApp()
    legacy = Mock(side_effect=AssertionError("Quit invoked legacy upload"))
    monkeypatch.setattr(app, "_send_traces_worker", legacy, raising=False)
    async with app.run_test(size=(100, 40)) as pilot:
        result = asyncio.create_task(asyncio.to_thread(app._request_trace_consent))
        await wait_for_consent(app, pilot)
        await app.action_quit()
        assert await asyncio.wait_for(result, timeout=5) is False
    assert app._trace_consent_pending is None
    legacy.assert_not_called()


def test_consent_callback_refuses_nonboolean_screen_results(monkeypatch):
    app = CrewRunApp()
    monkeypatch.setattr(app, "call_from_thread", lambda callback: callback())
    monkeypatch.setattr(app, "push_screen", lambda screen, callback: callback("yes"))
    monkeypatch.setattr(app, "_dismiss_consent_modal", lambda: None)
    assert app._request_trace_consent() is False


@pytest.mark.asyncio
async def test_conversation_quit_can_ask_for_deferred_session_consent(monkeypatch):
    app = ConsentApp(conversational=True)
    decisions = []

    class DeferredFlow:
        defer_trace_finalization = True

        def finalize_session_traces(self):
            decisions.append(app._request_trace_consent())

    app._flow = DeferredFlow()
    app._conversation_previous_defer_trace_finalization = False
    legacy = Mock(side_effect=AssertionError("Conversation quit invoked legacy upload"))
    monkeypatch.setattr(app, "_send_traces_worker", legacy, raising=False)
    async with app.run_test(size=(100, 40)) as pilot:
        quit_task = asyncio.create_task(app.action_quit())
        await wait_for_consent(app, pilot)
        await pilot.press("n")
        await asyncio.wait_for(quit_task, timeout=5)
    assert decisions == [False]
    assert app._flow.defer_trace_finalization is False
    assert app._trace_consent_pending is None
    legacy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_turn", [False, True])
async def test_quit_during_turn_discards_trace_after_worker_finishes(
    monkeypatch, prior_turn
):
    entered, release, finalized = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    app = ConsentApp(conversational=True)
    completions = []

    class RunningFlow:
        defer_trace_finalization = True
        trace = "prior turn" if prior_turn else None

        def handle_turn(self, message):
            entered.set()
            release.wait(timeout=5)
            # A first kickoff stores its deferred lifetime just before returning.
            self.trace = "completed turn"
            return "done"

        def finalize_session_traces(self, *, discard=False):
            completions.append(discard)
            self.trace = None
            finalized.set()

    app._flow = RunningFlow()
    app._conversation_previous_defer_trace_finalization = False
    app._conversation_turn_in_progress = True
    app._status = "working"
    prompt = Mock(side_effect=AssertionError("Cancelled execution asked for consent"))
    monkeypatch.setattr(app, "push_screen", prompt)
    async with app.run_test(size=(100, 40)):
        app._run_conversation_turn_worker("hello")
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            await app.action_quit()
            assert completions == []
            assert app._request_trace_consent() is False
        finally:
            release.set()
        assert await asyncio.to_thread(finalized.wait, 5)
    assert completions == [True]
    assert app._flow.trace is None
    assert app._flow.defer_trace_finalization is False
    prompt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["crew", "flow"])
async def test_quit_during_execution_rejects_late_consent(monkeypatch, kind):
    entered, release = threading.Event(), threading.Event()
    app = ConsentApp()
    decisions = []

    class RunningExecution:
        def kickoff(self, inputs=None):
            entered.set()
            release.wait(timeout=5)
            decisions.append(app._request_trace_consent())
            return "done"

    setattr(app, f"_{kind}", RunningExecution())
    app._status = "working"
    prompt = Mock(side_effect=AssertionError("Cancelled execution asked for consent"))
    completed = Mock()
    failed = Mock()
    monkeypatch.setattr(app, "push_screen", prompt)
    monkeypatch.setattr(app, "_on_crew_done", completed)
    monkeypatch.setattr(app, "_on_crew_failed", failed)
    async with app.run_test(size=(100, 40)):
        # Own the worker thread so the test awaits its body even after Textual
        # cancels its worker wrappers on exit (running threads are not stopped).
        run_worker = getattr(app, f"_run_{kind}_worker").__wrapped__
        worker = asyncio.create_task(asyncio.to_thread(run_worker, app))
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            await app.action_quit()
        finally:
            release.set()
        await asyncio.wait_for(worker, timeout=5)
    assert decisions == [False]
    assert app._trace_consent_pending is None
    prompt.assert_not_called()
    completed.assert_not_called()
    failed.assert_not_called()
