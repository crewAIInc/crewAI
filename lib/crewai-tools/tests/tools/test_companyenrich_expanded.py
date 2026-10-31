import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from crewai_tools import tools
from crewai_tools.generate_tool_specs import ToolSpecExtractor
from crewai_tools.tools.companyenrich_tool import CompanyEnrichAdvancedRequestTool
from crewai_tools.tools.companyenrich_tool._catalog import OPERATIONS

CASES = json.loads(
    Path(__file__).with_name("companyenrich_operations.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["operation"])
def test_every_advanced_operation_dispatches_once_losslessly(case):
    tool = CompanyEnrichAdvancedRequestTool(api_key="test-key")
    with patch(
        "crewai_tools.tools.companyenrich_tool.companyenrich_tool.requests.request"
    ) as send:
        send.return_value = Mock(status_code=200)
        send.return_value.json.return_value = case["result"]
        assert (
            tool.run(
                **{
                    k: case[k]
                    for k in (
                        "operation",
                        "body",
                        "query",
                        "path_parameters",
                        "allow_beta",
                    )
                }
            )
            == case["result"]
        )
    send.assert_called_once()
    args, kwargs = send.call_args
    assert args == (case["method"], "https://api.companyenrich.com" + case["path"])
    assert kwargs["allow_redirects"] is False
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


CORE = [
    (
        "CompanyEnrichCompanyTool",
        {
            "domain": "example.com",
            "expand": ["workforce"],
            "wait_for_enrichment": False,
        },
        "GET",
        "/companies/enrich",
    ),
    (
        "CompanyEnrichSearchTool",
        {
            "filters": {
                "countries": ["US"],
                "query": "literal",
                "future": {"nested": [None]},
            },
            "semantic_query": "software",
            "expand": ["workforce"],
        },
        "POST",
        "/companies/search",
    ),
    (
        "CompanyEnrichSimilarTool",
        {
            "domains": ["example.com"],
            "filters": {"minScore": 0.7},
            "expand": ["workforce"],
            "wait_for_enrichment": False,
        },
        "POST",
        "/companies/similar",
    ),
    (
        "CompanyEnrichPeopleSearchTool",
        {
            "filters": {"companyFilter": {"domains": ["example.com"]}},
            "expand": ["education"],
        },
        "POST",
        "/people/search",
    ),
    (
        "CompanyEnrichPersonLookupTool",
        {"email": "person@example.com", "expand": ["education"]},
        "POST",
        "/people/lookup",
    ),
    (
        "CompanyEnrichWorkforceTool",
        {"domain": "example.com"},
        "GET",
        "/companies/workforce",
    ),
    (
        "CompanyEnrichPropertiesTool",
        {"properties": {"linkedinId": "123"}},
        "POST",
        "/companies/enrich",
    ),
    (
        "CompanyEnrichBatchTool",
        {"domains": ["example.com"]},
        "POST",
        "/companies/enrich/batch",
    ),
    (
        "CompanyEnrichCompanyByIdTool",
        {"company_id": "123e4567-e89b-12d3-a456-426614174000"},
        "GET",
        "/companies",
    ),
    (
        "CompanyEnrichPersonByIdTool",
        {"person_id": "9007199254740993"},
        "GET",
        "/people",
    ),
    (
        "CompanyEnrichSearchCountTool",
        {"filters": {"countries": ["US"]}},
        "POST",
        "/companies/search/count",
    ),
    (
        "CompanyEnrichSimilarCountTool",
        {"domains": ["example.com"], "filters": {"minScore": 0.5}},
        "POST",
        "/companies/similar/count",
    ),
]


@pytest.mark.parametrize("name,inputs,method,path", CORE)
def test_core_catalog_tools_and_full_inputs(name, inputs, method, path):
    tool = getattr(tools, name)(api_key="test-key")
    result = (
        0
        if path.endswith("/count")
        else [{"full": {"new": None}}]
        if path.endswith("/batch")
        else {"full": {"new": None}}
    )
    with patch(
        "crewai_tools.tools.companyenrich_tool.companyenrich_tool.requests.request"
    ) as send:
        send.return_value = Mock(status_code=200)
        send.return_value.json.return_value = result
        assert tool.run(**inputs) == result
    args, kwargs = send.call_args
    assert args == (method, "https://api.companyenrich.com" + path)
    if "filters" in inputs:
        for key, value in inputs["filters"].items():
            assert kwargs["json"][key] == value
    if "expand" in inputs:
        assert kwargs["params"]["expand"] == inputs["expand"]
    if "wait_for_enrichment" in inputs:
        assert kwargs["params"]["waitForEnrichment"] == "false"


def test_all_core_and_advanced_tools_registered_with_schemas():
    specs = {x["name"]: x for x in ToolSpecExtractor().extract_all_tools()}
    for name in [x[0] for x in CORE] + ["CompanyEnrichAdvancedRequestTool"]:
        assert name in specs
        assert "api_key" not in specs[name]["run_params_schema"]["properties"]
    schema = CompanyEnrichAdvancedRequestTool().args_schema.model_json_schema()
    assert set(schema["$defs"]["CompanyEnrichOperation"]["enum"]) == set(OPERATIONS)


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("get_me", {}),
        ("post_companies_search_async", {"body": {"count": 1, "similar": {}}}),
        ("post_companies_similar_count", {"body": {"domain": "example.com"}}),
        (
            "post_companies_similar_count",
            {"body": {"domains": ["example.com"], "domain": "example.com"}},
        ),
        ("get_companies_enrich", {"query": {"domain": "", "Authorization": "x"}}),
        ("get_companies_workforce", {"query": {}}),
        (
            "get_companies_workforce",
            {
                "query": {
                    "domain": "example.com",
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                }
            },
        ),
        ("post_companies_enrich", {"body": {}}),
        ("post_companies_enrich_batch", {"body": {"domains": ["x"] * 51}}),
        ("post_companies_search", {"body": {"page": True}}),
        ("post_companies_search", {"body": {"page": 1001}}),
        ("post_companies_search", {"body": {"semanticQuery": "x" * 501}}),
        ("post_companies_similar", {"body": {"domains": []}}),
        ("post_people_lookup", {"body": {"email": " "}}),
        ("get_people", {"query": {"id": "9223372036854775808"}}),
        ("get_people_email", {"query": {"id": "1"}}),
        ("post_people_email_bulk", {"allow_beta": True, "body": {"items": []}}),
        ("post_companies_search_async", {"body": {}}),
        ("post_people_search_async", {"body": {"count": 50001}}),
        ("get_jobs_jobid", {"path_parameters": {"jobId": "../../me"}}),
    ],
)
def test_invalid_input_never_sends(operation, kwargs):
    with (
        patch(
            "crewai_tools.tools.companyenrich_tool.companyenrich_tool.requests.request"
        ) as send,
        pytest.raises((ValueError, TypeError)),
    ):
        CompanyEnrichAdvancedRequestTool(api_key="test-key").run(
            operation=operation, **kwargs
        )
    send.assert_not_called()


@pytest.mark.parametrize(
    "name,inputs",
    [
        ("CompanyEnrichSearchTool", {"filters": {"countries": ["US"]}, "page": True}),
        ("CompanyEnrichPeopleSearchTool", {"filters": {}, "page_size": True}),
        ("CompanyEnrichPersonByIdTool", {"person_id": True}),
        (
            "CompanyEnrichAdvancedRequestTool",
            {
                "operation": "get_people_email",
                "query": {"id": "1"},
                "allow_beta": "true",
            },
        ),
    ],
)
def test_registered_inputs_do_not_coerce_boolean_values(name, inputs):
    with (
        patch(
            "crewai_tools.tools.companyenrich_tool.companyenrich_tool.requests.request"
        ) as send,
        pytest.raises(ValueError),
    ):
        getattr(tools, name)(api_key="test-key").run(**inputs)
    send.assert_not_called()


def test_advanced_paid_error_sanitized_once():
    import requests

    request = requests.Request(
        "POST",
        "https://api.companyenrich.com",
        headers={"Authorization": "Bearer test-key"},
    ).prepare()
    error = requests.HTTPError("test-key upstream body", request=request)
    with (
        patch(
            "crewai_tools.tools.companyenrich_tool.companyenrich_tool.requests.request",
            side_effect=error,
        ) as send,
        pytest.raises(requests.HTTPError) as exc,
    ):
        CompanyEnrichAdvancedRequestTool(api_key="test-key").run(
            operation="post_companies_search_async", body={"count": 1}
        )
    send.assert_called_once()
    assert "test-key" not in str(exc.value)
    assert exc.value.request is None
    assert exc.value.response is None
