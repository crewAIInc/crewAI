import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import requests

from crewai_tools import MarkovianStampTool


SAMPLE_RECEIPT = {
    "ok": True,
    "stamp_id": 1,
    "merkle_root": "65ddefa2b8d3fb994f2a4037f9dd8278688138bf1c5eaa9cdb64c73c02663466",
    "data_hash": "b1000ed4a5bbf3bc3280c706e85374bec6ef5524ee877425b8454a8f5b73e1da",
    "block_height": 135213,
    "verify_url": "https://api.quantsynth.net/verify/65ddefa2b8d3fb994f2a4037f9dd8278688138bf1c5eaa9cdb64c73c02663466",
}


def test_run_uses_sdk_when_available(monkeypatch):
    tool = MarkovianStampTool()

    fake_client = MagicMock()
    fake_client.stamp.return_value = SAMPLE_RECEIPT
    fake_client_cls = MagicMock(return_value=fake_client)

    fake_module = ModuleType("markovian")
    fake_module.MarkovianClient = fake_client_cls
    monkeypatch.setitem(sys.modules, "markovian", fake_module)

    result = tool.run(data="hello world")

    fake_client_cls.assert_called_once()
    fake_client.stamp.assert_called_once()
    args, kwargs = fake_client.stamp.call_args
    assert args[0] == "hello world"
    assert SAMPLE_RECEIPT["merkle_root"] in result
    assert SAMPLE_RECEIPT["verify_url"] in result
    assert "135213" in result


def test_run_falls_back_to_post(monkeypatch):
    monkeypatch.delenv("MARKOVIAN_API_KEY", raising=False)
    tool = MarkovianStampTool()

    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_RECEIPT
    mock_resp.raise_for_status.return_value = None

    def fake_import(name, *args, **kwargs):
        if name == "markovian":
            raise ImportError("no markovian")
        return original_import(name, *args, **kwargs)

    import builtins

    original_import = builtins.__import__
    with patch("builtins.__import__", side_effect=fake_import):
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = tool.run(data="hello world")

    mock_post.assert_called_once()
    assert SAMPLE_RECEIPT["merkle_root"] in result


def test_run_handles_http_error():
    tool = MarkovianStampTool()
    err = requests.HTTPError()
    err.response = MagicMock(status_code=500, text="boom")

    with patch.object(tool, "_stamp", side_effect=err):
        result = tool.run(data="hello world")

    assert "Markovian stamp failed" in result
    assert "500" in result
    assert "boom" in result


def test_run_handles_non_dict_response():
    tool = MarkovianStampTool()
    with patch.object(tool, "_stamp", return_value=["not", "a", "dict"]):
        result = tool.run(data="hello world")

    assert "unexpected response" in result


def test_run_handles_missing_merkle_root():
    tool = MarkovianStampTool()
    with patch.object(tool, "_stamp", return_value={"ok": True}):
        result = tool.run(data="hello world")

    assert "no merkle_root" in result
