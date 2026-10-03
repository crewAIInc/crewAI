import asyncio
import hashlib
from unittest.mock import MagicMock, patch

import requests

from crewai_tools import MarkovianStampTool


DATA = "hello world"
DATA_HASH = hashlib.sha256(DATA.encode("utf-8")).hexdigest()

SAMPLE_RECEIPT = {
    "ok": True,
    "stamp_id": 4507,
    "log_index": 8330,
    "merkle_root": "11234ad23f343bd99b8ee173016e72a77eda97d432bc336f8187529d8c757b1e",
    "data_hash": DATA_HASH,
    "block_height": None,
    "verify_url": "https://api.markovianprotocol.com/verify/11234ad23f343bd99b8ee173016e72a77eda97d432bc336f8187529d8c757b1e",
}


def _ok_response():
    resp = MagicMock()
    resp.json.return_value = SAMPLE_RECEIPT
    resp.raise_for_status.return_value = None
    return resp


def test_run_sends_only_the_hash():
    tool = MarkovianStampTool()
    with patch("requests.post", return_value=_ok_response()) as mock_post:
        result = tool.run(data=DATA, label="demo")

    mock_post.assert_called_once()
    sent = mock_post.call_args.kwargs["json"]
    assert sent == {"data_hash": DATA_HASH, "label": "demo"}
    assert DATA not in repr(mock_post.call_args)
    assert mock_post.call_args.args[0] == "https://api.markovianprotocol.com/stamp"

    assert DATA_HASH in result
    assert SAMPLE_RECEIPT["merkle_root"] in result
    assert SAMPLE_RECEIPT["verify_url"] in result
    assert "8330" in result
    assert "block_height" not in result
    assert "None" not in result


def test_run_omits_log_index_when_absent():
    tool = MarkovianStampTool()
    receipt = {k: v for k, v in SAMPLE_RECEIPT.items() if k != "log_index"}
    with patch.object(tool, "_stamp", return_value=receipt):
        result = tool.run(data=DATA)

    assert "log_index" not in result
    assert SAMPLE_RECEIPT["merkle_root"] in result


def test_run_handles_http_error():
    tool = MarkovianStampTool()
    err = requests.HTTPError()
    err.response = MagicMock(status_code=500, text="boom")

    with patch.object(tool, "_stamp", side_effect=err):
        result = tool.run(data=DATA)

    assert "Markovian stamp failed" in result
    assert "500" in result
    assert "boom" in result


def test_run_handles_timeout():
    tool = MarkovianStampTool()
    with patch.object(tool, "_stamp", side_effect=requests.Timeout()):
        result = tool.run(data=DATA)

    assert "timed out" in result


def test_run_handles_non_dict_response():
    tool = MarkovianStampTool()
    with patch.object(tool, "_stamp", return_value=["not", "a", "dict"]):
        result = tool.run(data=DATA)

    assert "unexpected response" in result


def test_run_handles_missing_merkle_root():
    tool = MarkovianStampTool()
    with patch.object(tool, "_stamp", return_value={"ok": True}):
        result = tool.run(data=DATA)

    assert "no merkle_root" in result


def test_arun_stamps_off_the_event_loop():
    tool = MarkovianStampTool()
    with patch.object(tool, "_stamp", return_value=SAMPLE_RECEIPT):
        result = asyncio.run(tool._arun(data=DATA))

    assert SAMPLE_RECEIPT["merkle_root"] in result
