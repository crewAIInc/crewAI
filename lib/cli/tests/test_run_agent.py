import json
import subprocess
from types import SimpleNamespace

import click
import pytest

import crewai_cli.run_agent as runner
import crewai_cli.run_crew as dispatcher


def test_headless_chat_reports_no_terminal(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        dispatcher, "record_run_stopped", lambda reason, error=None: stopped.append(reason)
    )
    monkeypatch.setattr(runner, "is_interactive", lambda: False)
    with pytest.raises(click.UsageError, match="interactive terminal"):
        runner.run_agent_in_project_env(None)
    assert stopped == ["no_terminal"]


@pytest.mark.parametrize("error,reason,exception", [
    (FileNotFoundError("uv unavailable"), "launch_failed", click.ClickException),
    (PermissionError("launch denied"), "launch_failed", click.ClickException),
    (subprocess.CalledProcessError(7, ["uv"]), "script_failed", SystemExit),
    (subprocess.CalledProcessError(130, ["uv"]), None, SystemExit),
])
def test_agent_launch_failures_are_reported(monkeypatch, error, reason, exception):
    stopped = []
    monkeypatch.setattr(runner, "record_run_stopped", stopped.append)
    monkeypatch.setattr(runner, "_install_project_dependencies_if_needed", lambda: None)
    monkeypatch.setattr(runner, "build_env_with_all_tool_credentials", lambda: {})

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(exception) as caught:
        runner.run_agent_in_project_env('{"messages":"hello"}')
    assert stopped == ([reason] if reason else [])
    if isinstance(error, subprocess.CalledProcessError):
        assert caught.value.code == error.returncode


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


def test_bare_run_opens_conversation_tui(tmp_path, monkeypatch):
    import crewai.project

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "is_interactive", lambda: True)
    calls = []
    agent = SimpleNamespace(chat=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(crewai.project, "load_agent_project", lambda: agent)
    runner.run_agent()
    assert calls == [{"tui": True}]


def test_bare_run_needs_terminal_before_loading_project(monkeypatch):
    import crewai.project

    monkeypatch.setattr(runner, "is_interactive", lambda: False)
    monkeypatch.setattr(crewai.project, "load_agent_project", lambda: pytest.fail("loaded without terminal"))
    with pytest.raises(click.UsageError, match="--inputs"):
        runner.run_agent()


def test_bare_run_forwards_absent_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "is_interactive", lambda: True)
    monkeypatch.setattr(runner, "_install_project_dependencies_if_needed", lambda: None)
    monkeypatch.setattr(runner, "build_env_with_all_tool_credentials", lambda: {})
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append(kwargs))
    runner.run_agent_in_project_env(None)
    assert calls[0]["env"][runner._CREWAI_AGENT_INPUTS_ENV] == ""


@pytest.mark.asyncio
async def test_agent_chat_tui_runs_two_turns_with_history(monkeypatch):
    from crewai import Agent, BaseLLM
    from crewai_cli.crew_run_tui import CrewRunApp

    class EchoLLM(BaseLLM):
        def __init__(self):
            super().__init__(model="test")
            self.received = []

        def supports_function_calling(self):
            return False

        def call(self, messages, **kwargs):
            self.received.append(str(messages))
            return "Final Answer: Done"

        async def acall(self, messages, **kwargs):
            return self.call(messages, **kwargs)

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    llm = EchoLLM()
    agent = Agent(role="Assistant", goal="Help", backstory="Helpful", llm=llm, conversational=True)
    app = CrewRunApp(crew_name="Assistant", conversational=True)
    app._flow = agent.as_flow()
    async with app.run_test() as pilot:
        for index, message in enumerate(["first", "second"]):
            await pilot.click("#conversation-input")
            await pilot.press(*message, "enter")
            for _ in range(80):
                await pilot.pause(0.05)
                if len(app._conversation_messages) == (index + 1) * 2:
                    break
            assert len(app._conversation_messages) == (index + 1) * 2
    assert app._conversation_messages == [("user", "first"), ("assistant", "Done"), ("user", "second"), ("assistant", "Done")]
    assert "first" in llm.received[1]


@pytest.mark.parametrize("blank", [False, True])
@pytest.mark.parametrize("pasted", [True, False])
def test_missing_api_key_is_asked_for_before_chat(tmp_path, monkeypatch, pasted, blank):
    import os

    import crewai.project

    monkeypatch.chdir(tmp_path)
    if blank:
        # A blank key would fail while the model is built, before the prompt.
        monkeypatch.setenv("OPENAI_API_KEY", "")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(runner, "is_interactive", lambda: True)
    chats = []
    agent = SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai",
            model="gpt-6.1-sol",
            api_key=None,
            base_url="https://models.example/v1",
        ),
        chat=lambda **kwargs: chats.append(kwargs),
    )

    def load_agent_project():
        assert "OPENAI_API_KEY" not in os.environ
        return agent

    monkeypatch.setattr(crewai.project, "load_agent_project", load_agent_project)
    asked = []

    def ask(name, *_, llm_endpoints):
        asked.append((name, llm_endpoints["base_url"]))
        return pasted

    monkeypatch.setattr(runner, "ask_for_api_key", ask)
    if pasted:
        runner.run_agent()
        assert chats == [{"tui": True}]
    else:
        with pytest.raises(click.ClickException, match="OPENAI_API_KEY is not set"):
            runner.run_agent()
        assert chats == []
    assert asked == [("OPENAI_API_KEY", "https://models.example/v1")]


def test_aborting_the_key_prompt_ends_the_run_as_aborted(tmp_path, monkeypatch):
    import crewai.project

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(runner, "is_interactive", lambda: True)
    agent = SimpleNamespace(
        llm=SimpleNamespace(provider="openai", model="gpt-6.1-sol", api_key=None),
        chat=lambda **kwargs: pytest.fail("chat started after an abort"),
    )
    monkeypatch.setattr(crewai.project, "load_agent_project", lambda: agent)

    def abort(*args, **kwargs):
        raise click.Abort

    monkeypatch.setattr(runner, "ask_for_api_key", abort)
    with pytest.raises(click.Abort):
        runner.run_agent()


def test_project_bootstrap_ends_an_aborted_agent_run_as_aborted(tmp_path):
    import os
    import subprocess
    import sys

    package = tmp_path / "crewai_cli"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run_agent.py").write_text(
        "import click\n"
        "_ABORTED_EXIT_CODE = 130\n"
        "def run_agent(inputs):\n"
        "    raise click.Abort\n"
    )
    env = dict(os.environ)
    env[runner._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV] = str(package)
    env[runner._CREWAI_AGENT_INPUTS_ENV] = ""
    result = subprocess.run(
        [sys.executable, "-c", runner._AGENT_RUNNER_CODE],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == runner._ABORTED_EXIT_CODE
    assert "Aborted!" in result.stderr


def test_configured_api_key_is_not_asked_for(tmp_path, monkeypatch):
    import crewai.project

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(runner, "is_interactive", lambda: True)
    chats = []
    agent = SimpleNamespace(
        llm=SimpleNamespace(provider="openai", model="gpt-6.1-sol", api_key=None),
        chat=lambda **kwargs: chats.append(kwargs),
    )
    monkeypatch.setattr(crewai.project, "load_agent_project", lambda: agent)
    monkeypatch.setattr(
        runner, "ask_for_api_key", lambda *_: pytest.fail("asked for a set key")
    )
    runner.run_agent()
    assert chats == [{"tui": True}]
