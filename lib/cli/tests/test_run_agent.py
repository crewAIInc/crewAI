import json
from types import SimpleNamespace

import click
import pytest

import crewai_cli.run_agent as runner
import crewai_cli.run_crew as dispatcher


@pytest.mark.parametrize(
    "inputs",
    [
        None,
        "{}",
        '{"messages":" "}',
        '{"messages":[]}',
        '{"messages":42}',
        '{"messages":["hello"]}',
        '{"messages":[{"role":"user"}]}',
        '{"messages":[{"role":[],"content":"hello"}]}',
        '{"messages":[{"role":"user","content":["badpart"]}]}',
        '{"messages":"hello","typo":1}',
    ],
)
def test_agent_inputs_fail_before_launch(inputs, monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **kw: pytest.fail("launched invalid input")
    )
    with pytest.raises(click.UsageError):
        runner.run_agent_in_project_env(inputs)


def test_local_run_preserves_multimodal_messages_and_loads_dotenv(
    tmp_path, monkeypatch, capsys
):
    import os
    import crewai.project

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANAGED_AGENT_TEST_CONFIG", raising=False)
    (tmp_path / ".env").write_text("MANAGED_AGENT_TEST_CONFIG=from-project\n")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image.png"},
                },
            ],
        }
    ]
    received = []

    async def kickoff(**kwargs):
        received.append(kwargs["messages"])
        assert os.environ["MANAGED_AGENT_TEST_CONFIG"] == "from-project"
        return SimpleNamespace(raw="done")

    monkeypatch.setattr(
        crewai.project,
        "load_agent_project",
        lambda: SimpleNamespace(kickoff_async=kickoff),
    )
    runner.run_agent(json.dumps({"messages": messages}))
    assert received == [messages]
    assert capsys.readouterr().out == "done\n"


@pytest.mark.parametrize("manager", ["uv", "poetry"])
def test_agent_project_launches_current_cli_in_project_environment(
    tmp_path, monkeypatch, manager
):
    monkeypatch.chdir(tmp_path)
    if manager == "poetry":
        (tmp_path / "poetry.lock").write_text("# lock\n")
    monkeypatch.setattr(runner, "_install_project_dependencies_if_needed", lambda: None)
    monkeypatch.setattr(
        runner, "build_env_with_all_tool_credentials", lambda: {"KEEP": "value"}
    )
    calls = []
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    inputs = '{"messages":"hello"}'
    runner.run_agent_in_project_env(inputs)
    expected = (
        ["uv", "run", "--no-sync", "python", "-c"]
        if manager == "uv"
        else ["poetry", "run", "python", "-c"]
    )
    assert calls[0][0][0][:-1] == expected
    assert calls[0][1]["env"][runner._CREWAI_AGENT_INPUTS_ENV] == inputs
    assert calls[0][1]["env"][runner._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV].endswith(
        "crewai_cli"
    )
    assert calls[0][1]["env"]["KEEP"] == "value"


def test_agent_project_dispatch_and_crew_only_flag(monkeypatch):
    monkeypatch.setattr(
        dispatcher, "read_toml", lambda: {"tool": {"crewai": {"type": "agent"}}}
    )
    monkeypatch.setattr(dispatcher, "get_or_create_project_id", lambda: None)
    calls = []
    monkeypatch.setattr(runner, "run_agent_in_project_env", calls.append)
    dispatcher.run_crew(inputs='{"messages":"hello"}')
    assert calls == ['{"messages":"hello"}']
    with pytest.raises(click.UsageError, match="only be used when running crews"):
        dispatcher.run_crew(
            trained_agents_file="trained.pkl", inputs='{"messages":"hello"}'
        )


def test_agent_failure_is_nonzero(monkeypatch, tmp_path):
    import crewai.project

    monkeypatch.chdir(tmp_path)

    def fail():
        raise ValueError("invalid export")

    monkeypatch.setattr(crewai.project, "load_agent_project", fail)
    with pytest.raises(click.ClickException, match="Agent run failed: invalid export"):
        runner.run_agent('{"messages":"hello"}')


@pytest.mark.parametrize("failure", [False, True])
def test_project_bootstrap_dispatches_agent_with_unchanged_history(tmp_path, failure):
    import os
    import subprocess
    import sys

    package = tmp_path / "crewai_cli"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run_agent.py").write_text(
        "import click\n"
        "def run_agent(inputs):\n"
        + (
            "    raise click.ClickException('invalid config')\n"
            if failure
            else "    click.echo(inputs)\n"
        )
        + "def _run_json_crew(**kwargs):\n"
        "    raise AssertionError('wrong runtime')\n"
    )
    inputs = json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    )
    env = dict(os.environ)
    env[runner._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV] = str(package)
    env[runner._CREWAI_AGENT_INPUTS_ENV] = inputs
    result = subprocess.run(
        [sys.executable, "-c", runner._AGENT_RUNNER_CODE],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if failure:
        assert result.returncode == 1
        assert "invalid config" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == json.loads(inputs)


def test_crew_bootstrap_ignores_agent_inputs(tmp_path):
    import os
    import subprocess
    import sys

    package = tmp_path / "crewai_cli"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run_crew.py").write_text(
        "def _run_json_crew(**kwargs):\n    print('crew runtime')\n"
    )
    env = dict(os.environ)
    env[runner._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV] = str(package)
    env[runner._CREWAI_AGENT_INPUTS_ENV] = '{"messages":"unrelated inherited value"}'
    result = subprocess.run(
        [sys.executable, "-c", dispatcher._JSON_CREW_RUNNER_CODE],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout == "crew runtime\n"
