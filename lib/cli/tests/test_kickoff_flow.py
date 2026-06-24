from __future__ import annotations

import sys

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
