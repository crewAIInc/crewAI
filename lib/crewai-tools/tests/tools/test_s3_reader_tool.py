"""Tests for the S3 reader tool."""

import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from crewai_tools.aws.s3.reader_tool import S3ReaderTool


def _boto_modules(client: Mock) -> dict[str, ModuleType]:
    """Build minimal boto modules for exercising the lazy imports."""
    boto3 = ModuleType("boto3")
    boto3.client = Mock(return_value=client)  # type: ignore[attr-defined]

    botocore = ModuleType("botocore")
    exceptions = ModuleType("botocore.exceptions")

    class ClientError(Exception):
        pass

    exceptions.ClientError = ClientError  # type: ignore[attr-defined]
    botocore.exceptions = exceptions  # type: ignore[attr-defined]
    return {
        "boto3": boto3,
        "botocore": botocore,
        "botocore.exceptions": exceptions,
    }


def test_s3_reader_closes_response_body() -> None:
    """Release the streaming response after a successful read."""
    body = Mock()
    body.read.return_value = b"hello"
    client = Mock()
    client.get_object.return_value = {"Body": body}

    with patch.dict(sys.modules, _boto_modules(client)):
        result = S3ReaderTool()._run("s3://bucket/key.txt")

    assert result == "hello"
    body.close.assert_called_once_with()


def test_s3_reader_closes_response_body_after_decode_error() -> None:
    """Release the streaming response when UTF-8 decoding fails."""
    body = Mock()
    body.read.return_value = b"\xff"
    client = Mock()
    client.get_object.return_value = {"Body": body}

    with (
        patch.dict(sys.modules, _boto_modules(client)),
        pytest.raises(UnicodeDecodeError),
    ):
        S3ReaderTool()._run("s3://bucket/key.txt")

    body.close.assert_called_once_with()


def test_s3_reader_closes_response_body_after_read_error() -> None:
    """Release the streaming response when reading the body fails."""
    body = Mock()
    body.read.side_effect = OSError("connection reset")
    client = Mock()
    client.get_object.return_value = {"Body": body}

    with (
        patch.dict(sys.modules, _boto_modules(client)),
        pytest.raises(OSError, match="connection reset"),
    ):
        S3ReaderTool()._run("s3://bucket/key.txt")

    body.close.assert_called_once_with()
