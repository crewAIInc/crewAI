from copy import copy, deepcopy

from crewai.llm import LLM


def test_litellm_copy_preserves_timeout_and_context_window_size() -> None:
    llm = LLM(
        model="gpt-4o-mini",
        is_litellm=True,
        timeout=600,
        context_window_size=12_345,
    )

    shallow = copy(llm)
    deep = deepcopy(llm)

    assert (shallow.timeout, shallow.context_window_size) == (600, 12_345)
    assert (deep.timeout, deep.context_window_size) == (600, 12_345)

def test_litellm_copy_ignores_duplicate_limit_keys_in_additional_params() -> None:
    llm = LLM(
        model="gpt-4o-mini",
        is_litellm=True,
        timeout=600,
        context_window_size=12_345,
    )
    llm.additional_params.update(timeout=30, context_window_size=4_096)

    shallow = copy(llm)
    deep = deepcopy(llm)

    assert (shallow.timeout, shallow.context_window_size) == (600, 12_345)
    assert (deep.timeout, deep.context_window_size) == (600, 12_345)
