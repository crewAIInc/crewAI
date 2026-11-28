from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from crewai_tools.generate_tool_specs import ToolSpecExtractor
from crewai_tools.tools.dealport_tool import DealportTool, DealportToolInput
import pytest
import requests


def _mock_response(
    status_code: int = 200, payload: dict | None = None, text: str | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    if payload is not None:
        resp.text = json.dumps(payload)
        resp.json.return_value = payload
    else:
        resp.text = text or ""
        resp.json.side_effect = ValueError("not json")
    return resp


class TestDealportToolAuthAndConfig:
    def test_missing_api_key_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DP_API_KEY", raising=False)
        monkeypatch.delenv("DEALPORT_API_KEY", raising=False)
        tool = DealportTool()
        result = tool._run(action="preview_companies", state="OH")
        assert "DP_API_KEY" in result

    def test_explicit_api_key_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DP_API_KEY", "dp_test_from_env")
        tool = DealportTool(api_key="dp_test_explicit")
        with patch(
            "crewai_tools.tools.dealport_tool.dealport_tool.requests.request"
        ) as mock_req:
            mock_req.return_value = _mock_response(
                200, {"matching_count": 42, "sample_companies": []}
            )
            out = tool._run(
                action="preview_companies",
                state="oh",
                naics=["238220"],
                independent=True,
            )

        parsed = json.loads(out)
        assert parsed["matching_count"] == 42
        call_kwargs = mock_req.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer dp_test_explicit"
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["url"] == "https://app.dealport.com/api/v1/companies/preview"
        assert call_kwargs["json"] == {
            "state": "OH",
            "naics": ["238220"],
            "independent": True,
        }

    def test_dealport_api_key_env_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DP_API_KEY", raising=False)
        monkeypatch.setenv("DEALPORT_API_KEY", "dp_test_alt_env")
        tool = DealportTool()
        with patch(
            "crewai_tools.tools.dealport_tool.dealport_tool.requests.request"
        ) as mock_req:
            mock_req.return_value = _mock_response(200, {"matching_count": 7})
            tool._run(action="preview_companies", state="TX")
        assert (
            mock_req.call_args.kwargs["headers"]["Authorization"]
            == "Bearer dp_test_alt_env"
        )

    def test_api_key_excluded_from_serialization(self) -> None:
        tool = DealportTool(api_key="dp_test_secret")
        dumped = tool.model_dump()
        assert "api_key" not in dumped


class TestDealportToolEndpoints:
    def test_search_companies_encodes_query_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DP_API_KEY", "dp_test_key")
        tool = DealportTool()
        with patch(
            "crewai_tools.tools.dealport_tool.dealport_tool.requests.request"
        ) as mock_req:
            mock_req.return_value = _mock_response(
                200, {"companies": [{"id": "00000000-0000-0000-0000-000000000001"}]}
            )
            out = tool._run(
                action="search_companies",
                state="oh",
                cities=["Columbus", "Dayton"],
                naics=["238220"],
                min_employees=15,
                independent=True,
                limit=10,
            )

        assert "00000000-0000-0000-0000-000000000001" in out
        req_url = mock_req.call_args.kwargs["url"]
        assert req_url.startswith("https://app.dealport.com/api/v1/companies?")
        assert "state=OH" in req_url
        assert "cities=Columbus%2CDayton" in req_url
        assert "naics=238220" in req_url
        assert "min_employees=15" in req_url
        assert "independent=1" in req_url
        assert "limit=10" in req_url

    def test_search_buildings_validates_borough_and_builds_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DP_API_KEY", "dp_test_key")
        tool = DealportTool()

        missing = tool._run(action="search_buildings")
        assert "borough is required" in missing

        invalid = tool._run(action="search_buildings", borough="LA")
        assert "MN, BK, QN, BX, or SI" in invalid

        with patch(
            "crewai_tools.tools.dealport_tool.dealport_tool.requests.request"
        ) as mock_req:
            mock_req.return_value = _mock_response(
                200, {"buildings": [{"bbl": "1008350041"}]}
            )
            out = tool._run(
                action="search_buildings",
                borough="mn",
                min_units=20,
                loan_due_by=2027,
                limit=5,
            )
        assert "1008350041" in out
        req_url = mock_req.call_args.kwargs["url"]
        assert (
            req_url
            == "https://app.dealport.com/api/v1/buildings?borough=MN&min_units=20&loan_due_by=2027&limit=5"
        )

    def test_get_record_blocks_path_traversal_and_fetches_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DP_API_KEY", "dp_test_key")
        tool = DealportTool()

        assert "record_id is required" in tool._run(action="get_record")
        assert "Error:" in tool._run(action="get_record", record_id="../balance")
        assert "Error:" in tool._run(action="get_record", record_id="abc/def")

        with patch(
            "crewai_tools.tools.dealport_tool.dealport_tool.requests.request"
        ) as mock_req:
            mock_req.return_value = _mock_response(
                200, {"id": "1008350041", "record_type": "building"}
            )
            out = tool._run(action="get_record", record_id="1008350041")
        assert json.loads(out)["id"] == "1008350041"
        assert (
            mock_req.call_args.kwargs["url"]
            == "https://app.dealport.com/api/v1/records/1008350041"
        )

    def test_find_owner_posts_company_and_optional_person(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DP_API_KEY", "dp_test_key")
        tool = DealportTool()

        assert "company_id is required" in tool._run(action="find_owner")

        with patch(
            "crewai_tools.tools.dealport_tool.dealport_tool.requests.request"
        ) as mock_req:
            mock_req.return_value = _mock_response(
                200,
                {"hit": True, "owner": {"title": "President"}},
            )
            out = tool._run(
                action="find_owner",
                company_id="00000000-0000-0000-0000-000000000001",
                person="Alex Example",
            )
        assert json.loads(out)["hit"] is True
        assert (
            mock_req.call_args.kwargs["url"] == "https://app.dealport.com/api/v1/owners"
        )
        assert mock_req.call_args.kwargs["json"] == {
            "company_id": "00000000-0000-0000-0000-000000000001",
            "person": "Alex Example",
        }

    def test_http_error_and_timeout_handling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DP_API_KEY", "dp_test_key")
        tool = DealportTool()

        with patch(
            "crewai_tools.tools.dealport_tool.dealport_tool.requests.request"
        ) as mock_req:
            mock_req.return_value = _mock_response(
                402, {"code": "insufficient_balance"}
            )
            err_out = tool._run(action="search_companies", state="OH")
        assert "Dealport API error (402)" in err_out

        with patch(
            "crewai_tools.tools.dealport_tool.dealport_tool.requests.request",
            side_effect=requests.Timeout("timed out"),
        ):
            timeout_out = tool._run(action="preview_companies", state="OH")
        assert "timed out" in timeout_out

    def test_tool_spec_extractor_includes_dealport_tool(self) -> None:
        extractor = ToolSpecExtractor()
        extractor.extract_tool_info(DealportTool)
        assert len(extractor.tools_spec) == 1
        spec = extractor.tools_spec[0]
        assert spec["name"] == "DealportTool"
        assert spec["env_vars"][0]["name"] == "DP_API_KEY"
        assert "action" in spec["run_params_schema"]["properties"]
        assert DealportToolInput.model_json_schema()["properties"]["action"] is not None
