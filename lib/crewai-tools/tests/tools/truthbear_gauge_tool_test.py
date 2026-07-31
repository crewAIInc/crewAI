import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools import (
    TruthBearCatalogTool,
    TruthBearCoverageTool,
    TruthBearRecordTool,
)


def _ok(payload):
    """Mock a successful urlopen context manager returning `payload` as JSON."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    ctx = MagicMock()
    ctx.__enter__.return_value = response
    return ctx


def _http_error(status, payload):
    """Mock an HTTPError whose body is JSON - this is how a 402 challenge arrives."""
    return urllib.error.HTTPError(
        url="https://aeml-x402.zeabur.app/gauge",
        code=status,
        msg="Payment Required",
        hdrs=None,
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


COVERAGE_HIT = {
    "totals": {"signal_ids": 1, "distinct_entities": 10},
    "signals": [
        {
            "signal_id": "hydrology.river-level",
            "industry": "hydrology",
            "entities_count": 10,
            "freshness_counts": {"fresh": 10, "recent": 0, "stale": 0},
            "update_status": "on_schedule",
        }
    ],
}

COVERAGE_MISS = {"totals": {"signal_ids": 0, "distinct_entities": 0}, "signals": []}


@patch("urllib.request.urlopen")
def test_coverage_hit(mock_urlopen):
    mock_urlopen.return_value = _ok(COVERAGE_HIT)
    result = json.loads(TruthBearCoverageTool().run(signal_id="hydrology.river-level"))
    assert result["found"] is True
    assert result["signals"][0]["signal_id"] == "hydrology.river-level"
    assert result["signals"][0]["update_status"] == "on_schedule"


@patch("urllib.request.urlopen")
def test_coverage_miss_is_reported_not_raised(mock_urlopen):
    """An unknown signal comes back as HTTP 200 with an empty list, not an error.

    The tool must turn that into an explicit found:false, otherwise an agent can
    mistake an empty answer for a failed call.
    """
    mock_urlopen.return_value = _ok(COVERAGE_MISS)
    result = json.loads(TruthBearCoverageTool().run(signal_id="not.a.real.signal"))
    assert result["found"] is False
    assert result["signal_id"] == "not.a.real.signal"


@patch("urllib.request.urlopen")
def test_catalog_filtered(mock_urlopen):
    payload = {
        "signals": [
            {
                "signal_id": "hydrology.river-level",
                "entities": [
                    {"entity": "06893000", "name": "Missouri River at Kansas City, MO"}
                ],
            }
        ]
    }
    mock_urlopen.return_value = _ok(payload)
    result = json.loads(TruthBearCatalogTool().run(signal_id="hydrology.river-level"))
    assert result["signals"][0]["entities"][0]["entity"] == "06893000"


def test_catalog_requires_a_filter():
    """The unfiltered catalog is ~1.5 MB and would flood an agent's context."""
    with pytest.raises(ValueError):
        TruthBearCatalogTool().run()


@patch("urllib.request.urlopen")
def test_record_returns_402_challenge_intact(mock_urlopen):
    """402 is a price quote, not a failure - it must reach the caller unflattened."""
    challenge = {
        "x402Version": 1,
        "error": "X-PAYMENT header is required",
        "accepts": [
            {
                "scheme": "exact",
                "network": "base",
                "maxAmountRequired": "10000",
                "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "payTo": "0x2d16a243ba9fAcC6AC85519d5efad2149DC4C6c3",
            }
        ],
    }
    mock_urlopen.side_effect = _http_error(402, challenge)
    result = json.loads(
        TruthBearRecordTool().run(signal_id="hydrology.river-level", entity="06893000")
    )
    assert result["payment_required"] is True
    accepts = result["challenge"]["accepts"][0]
    assert accepts["network"] == "base"
    assert accepts["maxAmountRequired"] == "10000"
    assert accepts["payTo"].startswith("0x")
    # the tool must not have swallowed the challenge into a generic error string
    assert "error" not in result


@patch("urllib.request.urlopen")
def test_record_passes_through_a_paid_success(mock_urlopen):
    payload = {"signal_id": "hydrology.river-level", "record_hash": "a" * 64}
    mock_urlopen.return_value = _ok(payload)
    result = json.loads(
        TruthBearRecordTool().run(signal_id="hydrology.river-level", entity="06893000")
    )
    assert result["record_hash"] == "a" * 64
    assert "payment_required" not in result
