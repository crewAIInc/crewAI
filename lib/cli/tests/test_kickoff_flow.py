from __future__ import annotations

import sys

import pytest

from crewai_cli import kickoff_flow


def test_loads_conversational_flow_from_kickoff_script(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "src" / "demo_chat"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    (package_dir / "main.py").write_text(
        "\n".join(
            [
                "from crewai.flow import Flow",
                "",
                "class DemoChatFlow(Flow):",
                "    conversational = True",
            ]
        )
    )
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo-chat"',
                "[project.scripts]",
                'kickoff = "demo_chat.main:kickoff"',
            ]
        )
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("demo_chat.main", None)
    sys.modules.pop("demo_chat", None)

    flow = kickoff_flow._load_conversational_flow_from_kickoff_script()

    assert flow is not None
    assert type(flow).__name__ == "DemoChatFlow"
    assert flow.conversational is True


def test_kickoff_flow_falls_back_to_uv_when_no_conversational_flow(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, capture_output, text, check):
        calls.append(command)

        class Result:
            stderr = ""

        return Result()

    monkeypatch.setattr(
        kickoff_flow, "_load_conversational_flow_from_kickoff_script", lambda: None
    )
    monkeypatch.setattr(kickoff_flow.subprocess, "run", fake_run)

    kickoff_flow.kickoff_flow()

    assert calls == [["uv", "run", "kickoff"]]


def _patch_conversational_kickoff(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reached: list[str] = []
    monkeypatch.setattr(
        kickoff_flow, "_load_conversational_flow_from_kickoff_script", object
    )
    monkeypatch.setattr(
        kickoff_flow, "_run_conversational_flow_tui", lambda _flow: reached.append("tui")
    )
    monkeypatch.setattr(
        kickoff_flow.subprocess,
        "run",
        lambda *_args, **_kwargs: reached.append("uv"),
    )
    return reached


def test_kickoff_flow_conversational_flow_without_terminal_exits_with_guidance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reached = _patch_conversational_kickoff(monkeypatch)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        kickoff_flow.kickoff_flow()

    assert exc_info.value.code == 1
    assert reached == []
    err = capsys.readouterr().err
    assert "needs an interactive terminal" in err
    assert "flow.handle_turn(message, session_id=...)" in err


def test_kickoff_flow_conversational_flow_in_a_terminal_opens_the_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached = _patch_conversational_kickoff(monkeypatch)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    kickoff_flow.kickoff_flow()

    assert reached == ["tui"]


def test_run_conversational_flow_tui_initializes_event_listener(monkeypatch) -> None:
    calls: list[str] = []

    class FakeEventListener:
        def __init__(self) -> None:
            calls.append("listener")

    class FakeCrewRunApp:
        def __init__(self, *, crew_name: str, conversational: bool) -> None:
            calls.append("app")
            self.crew_name = crew_name
            self.conversational = conversational
            self._status = "completed"
            self._crew_result = "done"
            self._flow = None

        def run(self) -> None:
            calls.append("run")

    class DemoFlow:
        name = "Demo"

    monkeypatch.setattr(
        "crewai.events.event_listener.EventListener",
        FakeEventListener,
    )
    monkeypatch.setattr(
        "crewai_cli.crew_run_tui.CrewRunApp",
        FakeCrewRunApp,
    )

    result = kickoff_flow._run_conversational_flow_tui(DemoFlow())

    assert result == "done"
    assert calls == ["listener", "app", "run"]



def test_evaluate_on_an_untraced_conversation_runs_it_again_for_its_evaluation(
    monkeypatch,
) -> None:
    """The app closed because the session was not traced: the conversational
    flow runs again through `crewai eval`'s own path rather than the request
    being lost."""
    from types import SimpleNamespace

    from crewai_cli import kickoff_flow

    class FakeApp:
        def __init__(self, **kwargs):
            self._status = "completed"
            self._crew_result = "result"
            self._want_traced_rerun = True

        def run(self):
            pass

    monkeypatch.setattr("crewai_cli.crew_run_tui.CrewRunApp", FakeApp)
    monkeypatch.setattr("crewai.events.event_listener.EventListener", lambda: None)
    evaluated: list[dict[str, object]] = []
    monkeypatch.setattr(
        "crewai_cli.experimental.eval_crew.eval_crew",
        lambda **kwargs: evaluated.append(kwargs),
    )

    chat = SimpleNamespace(name="Chat")
    assert kickoff_flow._run_conversational_flow_tui(chat) == "result"
    # named by its definition (`crewai run --definition chat.yaml`): that flow again
    assert kickoff_flow._run_conversational_flow_tui(chat, definition="chat.yaml") == "result"
    assert [e["run_args"] for e in evaluated] == [{}, {"definition": "chat.yaml"}]
