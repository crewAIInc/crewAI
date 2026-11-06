from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.ocr_tool.ocr_tool import OCRTool


def test_local_image_data_uri_uses_file_media_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local PNG is announced as image/png in the data URI, not image/jpeg."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "picture.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    llm = MagicMock()
    llm.call.return_value = "extracted text"
    tool = OCRTool()
    tool.llm = llm

    assert tool._run(image_path_url="picture.png") == "extracted text"

    sent_url = llm.call.call_args.kwargs["messages"][1]["content"][0]["image_url"][
        "url"
    ]
    assert sent_url.startswith("data:image/png;base64,")
