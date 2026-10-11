"""Unit tests for the shared ``warn_if_truncated`` finish-reason helper.

Issue #7013: token-cap truncation was only surfaced by the Bedrock provider;
this helper is the single normalized warning used by every provider.
"""

import logging
import re

import pytest

from crewai.llms._finish_reason_utils import warn_if_truncated

TRUNCATED_RE = re.compile(r"truncated by token cap", re.IGNORECASE)


@pytest.mark.parametrize(
    ("provider", "finish_reason"),
    [
        ("openai", "length"),
        ("openai", "max_output_tokens"),
        ("azure", "length"),
        ("anthropic", "max_tokens"),
        ("bedrock", "max_tokens"),
        ("gemini", "MAX_TOKENS"),
    ],
)
def test_warns_on_truncation_finish_reasons(
    caplog: pytest.LogCaptureFixture, provider: str, finish_reason: str
) -> None:
    """Each provider's raw token-cap value maps to the one normalized warning."""
    with caplog.at_level(logging.WARNING):
        warned = warn_if_truncated(
            finish_reason, provider, log=logging.getLogger()
        )

    assert warned is True
    records = [r for r in caplog.records if TRUNCATED_RE.search(r.getMessage())]
    assert len(records) == 1
    message = records[0].getMessage()
    assert provider in message
    assert finish_reason in message
    assert records[0].levelno == logging.WARNING


@pytest.mark.parametrize(
    ("provider", "finish_reason"),
    [
        ("openai", "stop"),
        ("openai", "tool_calls"),
        ("openai", "completed"),
        # The Responses API reports ``status="incomplete"`` also for
        # non-token stops (e.g. ``content_filter``); only the resolved
        # ``incomplete_details.reason == "max_output_tokens"`` may warn.
        ("openai", "incomplete"),
        ("azure", "stop"),
        ("anthropic", "end_turn"),
        ("anthropic", "tool_use"),
        ("bedrock", "end_turn"),
        ("gemini", "STOP"),
        ("openai", "MAX_TOKENS"),  # right value, wrong provider vocabulary
        ("gemini", "length"),  # right value, wrong provider vocabulary
        ("some-other-provider", "length"),
    ],
)
def test_silent_on_non_truncation_finish_reasons(
    caplog: pytest.LogCaptureFixture, provider: str, finish_reason: str
) -> None:
    """Normal terminal reasons (and cross-provider mismatches) never warn."""
    with caplog.at_level(logging.WARNING):
        warned = warn_if_truncated(
            finish_reason, provider, log=logging.getLogger()
        )

    assert warned is False
    assert not [r for r in caplog.records if TRUNCATED_RE.search(r.getMessage())]


def test_silent_on_missing_finish_reason(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        warned = warn_if_truncated(None, "openai", log=logging.getLogger())

    assert warned is False
    assert caplog.records == []


def test_warning_includes_the_effective_token_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """#7013 asks for the limit in effect on the request, not just the reason."""
    with caplog.at_level(logging.WARNING):
        warned = warn_if_truncated(
            "length", "openai", log=logging.getLogger(), token_limit=4096
        )

    assert warned is True
    message = caplog.records[-1].getMessage()
    assert "token_limit=4096" in message


def test_warning_omits_the_limit_when_none_was_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Providers without an explicit cap must not fabricate ``token_limit=None``."""
    with caplog.at_level(logging.WARNING):
        warned = warn_if_truncated(
            "length", "openai", log=logging.getLogger(), token_limit=None
        )

    assert warned is True
    assert "token_limit=" not in caplog.records[-1].getMessage()


def test_logs_through_the_injected_logger(caplog: pytest.LogCaptureFixture) -> None:
    """The ``log`` keyword decides where the warning is emitted."""

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def warning(self, msg: str, *args: object) -> None:
            self.calls.append((msg, args))

    recorder = _Recorder()
    warned = warn_if_truncated("length", "openai", log=recorder)  # type: ignore[arg-type]

    assert warned is True
    assert len(recorder.calls) == 1
    msg, args = recorder.calls[0]
    assert "truncated by token cap" in msg
    assert args == ("openai", "length")
    assert caplog.records == []


def test_truncation_concern_adds_exactly_one_public_function() -> None:
    """Vidit's 2026-09-03 review of #7110: keep the truncation util to one
    public function (the module's other public callable is the pre-existing
    choices-shaped extractor, unrelated to this feature)."""
    import crewai.llms._finish_reason_utils as module

    public: list[str] = [
        name
        for name, value in vars(module).items()
        if callable(value)
        and not name.startswith("_")
        and getattr(value, "__module__", None) == module.__name__
    ]
    assert sorted(public) == [
        "extract_choices_finish_reason_and_id",
        "warn_if_truncated",
    ]


def test_str_mixin_enum_is_matched_and_rendered_by_wire_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Azure SDK passes ``CompletionsFinishReason`` (a ``str``-mixin enum);
    matching and log rendering must use the underlying wire value."""
    from enum import Enum

    class FakeCompletionsFinishReason(str, Enum):
        TOKEN_LIMIT_REACHED = "length"

    with caplog.at_level(logging.WARNING):
        warned = warn_if_truncated(
            FakeCompletionsFinishReason.TOKEN_LIMIT_REACHED,
            "azure",
            log=logging.getLogger(),
        )

    assert warned is True
    message = caplog.records[-1].getMessage()
    assert "finish_reason=length" in message
    assert "CompletionsFinishReason" not in message
