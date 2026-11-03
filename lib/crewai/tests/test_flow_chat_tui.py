from unittest.mock import Mock

import pytest

from crewai.flow import ConversationState, Flow


class ChatFlow(Flow[ConversationState]):
    conversational = True


def test_tui_chat_uses_shared_launcher_and_preserves_turn_options(monkeypatch):
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    launch = Mock()
    monkeypatch.setattr("crewai_cli.kickoff_flow._run_conversational_flow_tui", launch)
    flow = ChatFlow()
    flow.chat(
        tui=True,
        session_id="session-1",
        exit_commands=("bye",),
        defer_trace_finalization=False,
        input_files={"brief": "brief.txt"},
    )
    launch.assert_called_once_with(
        flow,
        session_id="session-1",
        exit_commands=("bye",),
        defer_trace_finalization=False,
        input_files={"brief": "brief.txt"},
    )


def test_tui_chat_requires_a_terminal(monkeypatch):
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: False)
    with pytest.raises(ValueError, match="interactive terminal"):
        ChatFlow().chat(tui=True)


@pytest.mark.parametrize(
    "options",
    [
        {"prompt": "User: "},
        {"assistant_prefix": "Bot: "},
        {"input_fn": lambda _: "quit"},
        {"output_fn": lambda _: None},
        {"skip_empty": False},
    ],
)
def test_tui_rejects_repl_only_options(monkeypatch, options):
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    with pytest.raises(ValueError, match="plain chat"):
        ChatFlow().chat(tui=True, **options)


def test_tui_chat_requires_conversational_opt_in():
    with pytest.raises(ValueError, match="conversational flows"):
        Flow().chat(tui=True)
