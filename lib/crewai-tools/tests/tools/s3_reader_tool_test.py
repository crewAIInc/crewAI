import sys
import types
from unittest.mock import MagicMock

import pytest

from crewai_tools.aws.s3.reader_tool import S3ReaderTool


class FakeBody:
    """S3 response body stand-in that records close() calls."""

    def __init__(self, payload):
        self._payload = payload
        self.closed = 0

    def read(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def close(self):
        self.closed += 1


@pytest.fixture
def s3_client(monkeypatch):
    """Install a fake boto3/botocore so the tool runs without the SDK installed."""
    client = MagicMock()
    boto3_module = types.ModuleType("boto3")
    boto3_module.client = MagicMock(return_value=client)

    botocore_module = types.ModuleType("botocore")
    exceptions_module = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        pass

    exceptions_module.ClientError = ClientError
    botocore_module.exceptions = exceptions_module

    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    monkeypatch.setitem(sys.modules, "botocore", botocore_module)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions_module)
    return client, ClientError


def test_s3_reader_closes_body_on_success(s3_client):
    client, _ = s3_client
    body = FakeBody(b"file contents")
    client.get_object.return_value = {"Body": body}

    result = S3ReaderTool()._run("s3://bucket/key.txt")

    assert result == "file contents"
    assert body.closed == 1


def test_s3_reader_closes_body_when_decode_fails(s3_client):
    client, _ = s3_client
    body = FakeBody(b"\xff\xfe not utf-8")
    client.get_object.return_value = {"Body": body}

    with pytest.raises(UnicodeDecodeError):
        S3ReaderTool()._run("s3://bucket/key.txt")

    assert body.closed == 1


def test_s3_reader_closes_body_when_read_fails(s3_client):
    client, _ = s3_client
    body = FakeBody(OSError("connection reset"))
    client.get_object.return_value = {"Body": body}

    with pytest.raises(OSError):
        S3ReaderTool()._run("s3://bucket/key.txt")

    assert body.closed == 1


def test_s3_reader_keeps_client_error_behavior(s3_client):
    client, ClientError = s3_client
    client.get_object.side_effect = ClientError("NoSuchKey")

    result = S3ReaderTool()._run("s3://bucket/key.txt")

    assert result.startswith("Error reading file from S3:")
