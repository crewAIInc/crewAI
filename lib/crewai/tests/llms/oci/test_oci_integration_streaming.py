from __future__ import annotations


def test_oci_live_streaming_call(
    oci_chat_model: str,
    oci_live_llm_factory,
    oci_prompts: dict[str, str],
    oci_temperature_for_model,
    oci_token_budget,
):
    llm = oci_live_llm_factory(
        oci_chat_model,
        max_tokens=oci_token_budget(oci_chat_model, "stream"),
        temperature=oci_temperature_for_model(oci_chat_model),
        stream=True,
    )

    result = llm.call(oci_prompts["stream"])

    assert isinstance(result, str)
    assert result.strip()
