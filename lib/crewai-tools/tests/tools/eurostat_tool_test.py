from unittest.mock import MagicMock, patch

from crewai_tools import EurostatTool
import pytest
import requests


@pytest.fixture
def tool():
    return EurostatTool()


def _jsonstat_payload():
    """A small, hand-built JSON-stat payload: unemployment rate for DE, two months.

    Dimensions (in `id` order): geo (1 value), time (2 values). Row-major flat
    index means index 0 -> (geo[0], time[0]), index 1 -> (geo[0], time[1]).
    """
    return {
        "label": "Unemployment rate",
        "id": ["geo", "time"],
        "size": [1, 2],
        "dimension": {
            "geo": {"category": {"index": {"DE": 0}, "label": {"DE": "Germany"}}},
            "time": {
                "category": {
                    "index": {"2026-01": 0, "2026-02": 1},
                    "label": {"2026-01": "2026-01", "2026-02": "2026-02"},
                }
            },
        },
        "value": {"0": 3.9, "1": 4.0},
    }


def _mock_response(json_data, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response


def _jsonstat_payload_two_by_two():
    """A JSON-stat payload where every dimension has size > 1.

    Unlike `_jsonstat_payload` (geo size 1), this actually exercises the
    row-major, last-dimension-fastest decoding in `_parse_jsonstat`: a bug
    that swapped axis order or fastest/slowest dimension would misassign
    labels here, whereas it couldn't show up with a size-1 axis.
    """
    return {
        "label": "Unemployment rate",
        "id": ["geo", "time"],
        "size": [2, 2],
        "dimension": {
            "geo": {"category": {"index": {"DE": 0, "FR": 1}, "label": {"DE": "Germany", "FR": "France"}}},
            "time": {
                "category": {
                    "index": {"2026-01": 0, "2026-02": 1},
                    "label": {"2026-01": "2026-01", "2026-02": "2026-02"},
                }
            },
        },
        # Row-major, time (last dim) fastest-varying: flat index = geo_pos * 2 + time_pos.
        "value": {"0": 10.0, "1": 20.0, "2": 30.0, "3": 40.0},
    }


def test_parse_jsonstat_orders_axes_correctly_with_nontrivial_dimensions(tool):
    parsed = tool._parse_jsonstat(_jsonstat_payload_two_by_two())
    by_key = {(r["geo"], r["time"]): r["value"] for r in parsed["records"]}

    assert by_key[("Germany", "2026-01")] == 10.0
    assert by_key[("Germany", "2026-02")] == 20.0
    assert by_key[("France", "2026-01")] == 30.0
    assert by_key[("France", "2026-02")] == 40.0


def test_run_requires_indicator_or_dataset_code(tool):
    result = tool._run()
    assert "Error" in result
    assert "indicator" in result


def test_run_rejects_both_indicator_and_dataset_code(tool):
    result = tool._run(indicator="unemployment_rate", dataset_code="une_rt_m")
    assert "Error" in result
    assert "exactly one" in result


@patch("crewai_tools.tools.eurostat_tool.eurostat_tool.requests.get")
def test_run_with_known_indicator(mock_get, tool):
    mock_get.return_value = _mock_response(_jsonstat_payload())

    result = tool._run(indicator="unemployment_rate", geo="DE")

    call_url = mock_get.call_args.args[0]
    params = mock_get.call_args.kwargs["params"]
    assert call_url.endswith("/une_rt_m")
    assert params["geo"] == "DE"
    assert params["s_adj"] == "SA"
    assert "Germany" in result
    assert "3.9" in result
    assert "4.0" in result


def test_run_with_unknown_indicator(tool):
    result = tool._run(indicator="not_a_real_indicator")
    assert "Error" in result
    assert "unknown indicator" in result.lower()


@patch("crewai_tools.tools.eurostat_tool.eurostat_tool.requests.get")
def test_run_with_raw_dataset_code_and_filters(mock_get, tool):
    mock_get.return_value = _mock_response(_jsonstat_payload())

    result = tool._run(
        dataset_code="une_rt_m", geo="DE", since="2026-01", filters={"sex": "T"}
    )

    params = mock_get.call_args.kwargs["params"]
    assert params["geo"] == "DE"
    assert params["sinceTimePeriod"] == "2026-01"
    assert params["sex"] == "T"
    assert "Germany" in result


@patch("crewai_tools.tools.eurostat_tool.eurostat_tool.requests.get")
def test_run_empty_result(mock_get, tool):
    empty_payload = {
        "label": "Empty",
        "id": [],
        "size": [],
        "value": {},
        "dimension": {},
    }
    mock_get.return_value = _mock_response(empty_payload)

    result = tool._run(dataset_code="some_empty_dataset")
    assert "No data found" in result


@patch("crewai_tools.tools.eurostat_tool.eurostat_tool.requests.get")
def test_run_async_extraction_warning(mock_get, tool):
    """A successful response with warning.status == 413 means 'still preparing', not 'no data'."""
    payload = {**_jsonstat_payload(), "warning": {"status": 413, "label": "too big"}}
    mock_get.return_value = _mock_response(payload)

    result = tool._run(dataset_code="une_rt_m")
    assert "preparing a large extraction" in result


@patch("crewai_tools.tools.eurostat_tool.eurostat_tool.requests.get")
def test_run_http_413_error(mock_get, tool):
    """An actual HTTP 413 means the extraction itself is too large, not a transient retry."""
    response = MagicMock()
    response.status_code = 413
    http_error = requests.exceptions.HTTPError(response=response)
    mock_get.return_value.raise_for_status.side_effect = http_error

    result = tool._run(dataset_code="une_rt_m")
    assert "too large" in result
    assert "Add filters" in result


@patch("crewai_tools.tools.eurostat_tool.eurostat_tool.requests.get")
def test_run_http_404_error(mock_get, tool):
    response = MagicMock()
    response.status_code = 404
    http_error = requests.exceptions.HTTPError(response=response)
    mock_get.return_value.raise_for_status.side_effect = http_error

    result = tool._run(dataset_code="not_a_real_dataset")
    assert "not found" in result.lower()


@patch("crewai_tools.tools.eurostat_tool.eurostat_tool.requests.get")
def test_run_generic_exception(mock_get, tool):
    mock_get.side_effect = Exception("Connection error")

    result = tool._run(dataset_code="une_rt_m")
    assert "Error" in result
    assert "Connection error" in result
