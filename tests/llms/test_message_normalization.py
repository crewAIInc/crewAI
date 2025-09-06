import copy
import pytest

from crewai.llm import LLM


@pytest.fixture
def base_messages():
    # a minimal, mixed-role conversation to mutate in tests (no user yet)
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "Got it."},
    ]


def _mk_llm(model_name: str) -> LLM:
    return LLM(model=model_name)


# ---------- o1 special-case ----------

def test_o1_rewrites_system_to_assistant(base_messages):
    llm = _mk_llm("openai/o1-mini")
    msgs = llm._format_messages_for_provider(copy.deepcopy(base_messages))

    # all system roles should be rewritten to assistant for o1
    assert all(m["role"] != "system" for m in msgs)
    # original system message should now be assistant
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "You are helpful."
    # rest preserved
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Got it."


def test_o1_does_not_touch_non_system(base_messages):
    llm = _mk_llm("openai/o1")
    msgs = llm._format_messages_for_provider(copy.deepcopy(base_messages))
    # assistant message remains assistant
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Got it."


# ---------- Gemini special-case ----------

def test_gemini_appends_user_when_missing(base_messages):
    # No user turns in base_messages -> Gemini should append one (not prepend)
    llm = _mk_llm("gemini-1.5-pro")
    msgs = llm._format_messages_for_provider(copy.deepcopy(base_messages))

    # first message remains system (no rewrite)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."

    # at least one user turn must be present
    assert any(m["role"] == "user" for m in msgs)

    # contract: if missing, Gemini appends a user turn
    assert msgs[-1]["role"] == "user"
    assert isinstance(msgs[-1]["content"], str)


def test_gemini_leaves_sequence_when_user_already_present_later():
    # If a user message already exists anywhere, do not inject another or reorder
    llm = _mk_llm("google/gemini-1.5-flash")
    messages = [
        {"role": "system", "content": "System instr."},
        {"role": "assistant", "content": "Ack."},
        {"role": "user", "content": "Now a user speaks."},
    ]
    msgs = llm._format_messages_for_provider(copy.deepcopy(messages))

    # Should be identical (no injected user, no rewrites)
    assert msgs == messages


def test_gemini_leaves_when_user_present_first():
    llm = _mk_llm("gemini-1.5-pro")
    messages = [
        {"role": "user", "content": "Start with a user turn."},
        {"role": "system", "content": "System instr."},
    ]
    msgs = llm._format_messages_for_provider(copy.deepcopy(messages))

    # Should be identical (no injected user, no rewrites)
    assert msgs == messages


# ---------- Other models (no-op) ----------

def test_other_models_unchanged_when_not_o1_or_gemini(base_messages):
    llm = _mk_llm("openai/gpt-4o")
    original = copy.deepcopy(base_messages)
    msgs = llm._format_messages_for_provider(copy.deepcopy(base_messages))
    assert msgs == original
