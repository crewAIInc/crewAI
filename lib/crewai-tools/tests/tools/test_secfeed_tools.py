import os
from unittest.mock import MagicMock, patch

from crewai_tools import (
    SecFeed8KReportsTool,
    SecFeedFilingSearchTool,
    SecFeedFinancialStatementsTool,
    SecFeedInsiderTradesTool,
)
from crewai_tools.tools.secfeed_tool.secfeed_tools import (
    best_passages,
    money,
    pick_section,
)
import pytest


INDEX = "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/0000000001-26-000001-index.htm"
FILING = {
    "accessionNo": "0000000001-26-000001",
    "cik": "1",
    "ticker": "ACME",
    "companyName": "ACME CORP",
    "formType": "10-K",
    "filedAt": "2026-02-25",
    "linkToFilingDetails": INDEX,
}


@pytest.fixture(autouse=True)
def secfeed_api_key():
    with patch.dict(os.environ, {"SECFEED_API_KEY": "sf_test"}):
        yield


def _response(json_data=None, text=None):
    response = MagicMock()
    response.json.return_value = json_data
    response.text = text
    response.raise_for_status.return_value = None
    return response


def test_pick_section():
    assert pick_section("What are the biggest risks?") == "risk"
    assert pick_section("Who are its main competitors?") == "business"
    assert pick_section("How exposed is it to interest rate changes?") == "market_risk"
    assert pick_section("How did revenue grow?") == "mdna"


def test_best_passages_keeps_matches_in_document_order():
    filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor. " * 3
    text = "\n\n".join(
        [
            filler,
            "Export controls on China reduced data center revenue this quarter. " * 2,
            filler,
            "Tariffs and export controls may change again and hurt China sales. " * 2,
        ]
    )
    out = best_passages(text, "export controls China", max_chars=400)
    assert "reduced data center" in out
    assert out.index("reduced") < out.index("Tariffs")
    assert "Lorem" not in out


def test_money():
    assert money("215938000000") == "215.94B"
    assert money(5000) == "5.00K"


def test_missing_api_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="SECFEED_API_KEY"):
            SecFeedFilingSearchTool()._key()


@patch("requests.request")
def test_filing_search_reads_risk_section_and_caches(mock_request):
    section = "Our risk factors.\n\n" + ("Export controls could reduce our sales in China. " * 3)
    mock_request.side_effect = [
        _response({"filings": [FILING]}),
        _response(text=section),
    ]
    tool = SecFeedFilingSearchTool()

    first = tool.run(ticker="acme", question="What are the export control risks?")
    second = tool.run(ticker="ACME", question="Any other risks in China?")

    assert "ACME CORP (ACME) 10-K filed 2026-02-25, section: Risk Factors" in first
    assert INDEX in first
    assert "Export controls could reduce" in second
    assert mock_request.call_count == 2  # filing and section fetched once
    method, url = mock_request.call_args_list[1].args
    assert (method, url) == ("GET", "https://secfeedapi.com/v1/extractor")
    assert mock_request.call_args_list[1].kwargs["params"]["item"] == "1A"
    assert mock_request.call_args_list[0].kwargs["headers"] == {"Authorization": "Bearer sf_test"}


@patch("requests.request")
def test_filing_search_no_filing(mock_request):
    mock_request.return_value = _response({"filings": []})
    out = SecFeedFilingSearchTool().run(ticker="NOPE", question="revenue", form="10-Q")
    assert out == "No 10-Q found for NOPE. Check the ticker."


@patch("requests.request")
def test_financial_statements(mock_request):
    mock_request.return_value = _response(
        [
            {
                "accessionNo": "0000000001-26-000001",
                "StatementsOfIncome": {
                    "Revenues": [
                        {"value": "215938000000", "period": {"endDate": "2026-01-25"}},
                        {"value": "130497000000", "period": {"endDate": "2025-01-26"}},
                    ]
                },
            }
        ]
    )
    out = SecFeedFinancialStatementsTool().run(ticker="acme")
    assert "Revenues: 2026-01-25: 215.94B; 2025-01-26: 130.50B" in out


@patch("requests.request")
def test_insider_trades_ignores_bad_code(mock_request):
    mock_request.return_value = _response(
        {
            "total": 1,
            "transactions": [
                {
                    "tx_date": "2026-11-10",
                    "code": "S",
                    "shares": 5000.0,
                    "price": 531.0,
                    "aff10b5one": True,
                    "owners": [{"name": "DOE JANE", "officerTitle": "CFO"}],
                }
            ],
        }
    )
    out = SecFeedInsiderTradesTool().run(ticker="ACME", code="", limit=5)
    assert "open market sale 5.00K sh @ 531.0 = $2.65M, 10b5-1 plan" in out
    assert mock_request.call_args.kwargs["params"] == {"ticker": "ACME", "size": 5}


@patch("requests.request")
def test_8k_reports(mock_request):
    mock_request.side_effect = [
        _response({"filings": [dict(FILING, formType="8-K")]}),
        _response(
            {
                "filings": [
                    {
                        "filed": "2026-08-26",
                        "items": [{"item": "2.02", "title": "Results of Operations and Financial Condition"}],
                        "url": INDEX,
                    }
                ]
            }
        ),
    ]
    out = SecFeed8KReportsTool().run(ticker="ACME", limit=3)
    assert "- 2026-08-26: 2.02 Results of Operations and Financial Condition" in out
    assert mock_request.call_args.kwargs["params"] == {"cik": "1", "size": 3}
