from __future__ import annotations

from pydantic import BaseModel


class OCIStructuredResponse(BaseModel):
    summary: str
    topic: str


def test_oci_live_structured_output(
    oci_chat_model: str,
    oci_live_llm_factory,
    oci_prompts: dict[str, str],
    oci_temperature_for_model,
    oci_token_budget,
):
    llm = oci_live_llm_factory(
        oci_chat_model,
        max_tokens=oci_token_budget(oci_chat_model, "structured"),
        temperature=oci_temperature_for_model(oci_chat_model),
    )

    result = llm.call(
        oci_prompts["structured"],
        response_model=OCIStructuredResponse,
    )

    assert isinstance(result, OCIStructuredResponse)
    assert result.summary.strip()
    assert result.topic.strip()
