"""Live integration tests for OCI Generative AI structured output.

Run with:
    OCI_AUTH_TYPE=API_KEY OCI_AUTH_PROFILE=API_KEY_AUTH \
    OCI_COMPARTMENT_ID=<compartment> OCI_REGION=us-chicago-1 \
    OCI_TEST_MODELS="meta.llama-3.3-70b-instruct" \
    uv run pytest tests/llms/oci/test_oci_integration_structured.py -v
"""

from __future__ import annotations

from pydantic import BaseModel

from crewai.llms.providers.oci.completion import OCICompletion


class CapitalResponse(BaseModel):
    """Response containing a country's capital city."""
    country: str
    capital: str


def test_oci_live_structured_output(oci_chat_model: str, oci_live_config: dict):
    """Structured output should return a validated Pydantic model."""
    llm = OCICompletion(model=oci_chat_model, **oci_live_config)
    result = llm.call(
        messages=[{"role": "user", "content": "What is the capital of France? Answer with country and capital fields."}],
        response_model=CapitalResponse,
    )

    assert isinstance(result, CapitalResponse)
    assert result.capital.lower() == "paris"
    assert result.country.lower() == "france"
