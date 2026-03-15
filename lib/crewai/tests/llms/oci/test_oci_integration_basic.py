from __future__ import annotations

import asyncio


def test_oci_live_basic_call(
    oci_chat_model: str,
    oci_live_llm_factory,
    oci_prompts: dict[str, str],
    oci_temperature_for_model,
    oci_token_budget,
):
    llm = oci_live_llm_factory(
        oci_chat_model,
        max_tokens=oci_token_budget(oci_chat_model, "basic"),
        temperature=oci_temperature_for_model(oci_chat_model),
    )

    result = llm.call(oci_prompts["basic"])

    assert isinstance(result, str)
    assert result.strip()


def test_oci_live_async_call(
    oci_chat_model: str,
    oci_live_llm_factory,
    oci_prompts: dict[str, str],
    oci_temperature_for_model,
    oci_token_budget,
):
    llm = oci_live_llm_factory(
        oci_chat_model,
        max_tokens=oci_token_budget(oci_chat_model, "async"),
        temperature=oci_temperature_for_model(oci_chat_model),
    )

    result = asyncio.run(llm.acall(oci_prompts["async"]))

    assert isinstance(result, str)
    assert result.strip()
