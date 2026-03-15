from __future__ import annotations

import asyncio


async def _collect_stream(llm, prompt: str) -> str:
    chunks: list[str] = []
    async for chunk in llm.astream(prompt):
        chunks.append(chunk)
    return "".join(chunks)


def test_oci_live_abatch(
    oci_chat_model: str,
    oci_live_llm_factory,
    oci_temperature_for_model,
    oci_token_budget,
):
    llm = oci_live_llm_factory(
        oci_chat_model,
        max_tokens=oci_token_budget(oci_chat_model, "async"),
        temperature=oci_temperature_for_model(oci_chat_model),
    )

    results = asyncio.run(
        llm.abatch(
            [
                "Reply with one short sentence about Oracle Cloud.",
                "Reply with one short sentence about databases.",
            ]
        )
    )

    assert len(results) == 2
    assert all(isinstance(result, str) and result.strip() for result in results)


def test_oci_live_astream(
    oci_chat_model: str,
    oci_live_llm_factory,
    oci_temperature_for_model,
    oci_token_budget,
):
    llm = oci_live_llm_factory(
        oci_chat_model,
        max_tokens=oci_token_budget(oci_chat_model, "stream"),
        temperature=oci_temperature_for_model(oci_chat_model),
    )

    result = asyncio.run(
        _collect_stream(llm, "Reply with a short sentence about Oracle Cloud.")
    )

    assert isinstance(result, str)
    assert result.strip()
