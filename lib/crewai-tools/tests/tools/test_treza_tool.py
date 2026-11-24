"""Tests for the Treza tools, against a scripted stand-in for Treza's API."""

from __future__ import annotations

import json
from typing import Any

import pytest

from crewai_tools.tools.treza_tool.treza_client import TrezaClient
from crewai_tools.tools.treza_tool.treza_tool import (
    TrezaGetRunTool,
    TrezaImageTool,
    TrezaVideoTool,
)


class _FakeTreza(TrezaClient):
    """Answers Treza tool calls from a script and records what was asked."""

    def __init__(self, answers: dict[str, list[tuple[int, dict[str, Any]]]]):
        super().__init__("test-key")
        self.answers = answers
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def request(
        self, tool: str, args: dict[str, Any] | None = None
    ) -> tuple[bool, int, dict[str, Any]]:
        self.calls.append((tool, args or {}))
        status, body = self.answers[tool].pop(0)
        return status < 400, status, body


QUOTE = {
    "pipelineId": "pipeline_1",
    "estimatedChargeUsd": 0.28,
    "sufficient": True,
    "quoteId": "q1",
    "model": {"id": "google/veo-3.1-lite"},
}
DONE = {
    "status": "success",
    "outputs": {"video": "https://media.trezalabs.com/clip.mp4"},
    "durationMs": 61000,
}


def _tool(cls: type, fake: _FakeTreza, **kwargs: Any) -> Any:
    tool = cls(api_key="test-key", **kwargs)
    tool._client = fake
    return tool


def test_video_quotes_starts_once_and_returns_the_clip() -> None:
    fake = _FakeTreza(
        {
            "make_video": [(200, QUOTE)],
            "run_pipeline": [(200, {"runId": "run_1"})],
            "get_run": [(200, DONE)],
        }
    )
    out = json.loads(
        _tool(TrezaVideoTool, fake, max_cost_usd=1).run(
            prompt="a lighthouse at dawn", aspect_ratio="9:16"
        )
    )

    assert out["url"] == "https://media.trezalabs.com/clip.mp4"
    assert out["price_usd"] == 0.28
    assert out["model"] == "google/veo-3.1-lite"
    assert out["run_id"] == "run_1"
    assert [c[0] for c in fake.calls] == ["make_video", "run_pipeline", "get_run"]
    assert fake.calls[0][1]["aspectRatio"] == "9:16"
    assert fake.calls[0][1]["maxPriceUsd"] == 1
    assert fake.calls[1][1] == {"pipelineId": "pipeline_1", "quoteId": "q1"}


def test_over_the_cap_spends_nothing() -> None:
    fake = _FakeTreza({"make_video": [(200, {**QUOTE, "estimatedChargeUsd": 2.63})]})
    out = _tool(TrezaVideoTool, fake, max_cost_usd=1).run(prompt="a long clip")

    assert out.startswith("Error: This render costs $2.63, over the max cost of $1.00")
    assert [c[0] for c in fake.calls] == ["make_video"]


def test_low_balance_spends_nothing() -> None:
    fake = _FakeTreza(
        {"make_image": [(200, {**QUOTE, "sufficient": False, "balanceUsd": 0.01})]}
    )
    out = _tool(TrezaImageTool, fake).run(prompt="a red apple")

    assert "Nothing was spent" in out
    assert [c[0] for c in fake.calls] == ["make_image"]


def test_no_cap_sends_no_price_limit() -> None:
    fake = _FakeTreza(
        {
            "make_image": [(200, {**QUOTE, "estimatedChargeUsd": 0})],
            "run_pipeline": [(200, {"runId": "run_2"})],
            "get_run": [
                (200, {"status": "success", "outputs": {"image": "https://x/i.png"}})
            ],
        }
    )
    out = json.loads(_tool(TrezaImageTool, fake).run(prompt="a red apple"))

    assert out["url"] == "https://x/i.png"
    assert "maxPriceUsd" not in fake.calls[0][1]


def test_a_render_still_going_returns_its_ids() -> None:
    fake = _FakeTreza(
        {
            "make_video": [(200, QUOTE)],
            "run_pipeline": [(200, {"runId": "run_3"})],
            "get_run": [(200, {"status": "running", "pollAfterSec": 5})],
        }
    )
    out = json.loads(_tool(TrezaVideoTool, fake, wait_seconds=0).run(prompt="waves"))

    assert out["status"] == "running"
    assert out["finished"] is False
    assert (out["run_id"], out["pipeline_id"]) == ("run_3", "pipeline_1")


def test_get_run_reports_why_a_render_stopped() -> None:
    fake = _FakeTreza(
        {"get_run": [(200, {"status": "error", "statusNote": "The model refused."})]}
    )
    out = _tool(TrezaGetRunTool, fake).run(pipeline_id="pipeline_1", run_id="run_4")

    assert out.startswith("Error: The render stopped before it finished.")
    assert "The model refused." in out


def test_a_bad_key_is_explained() -> None:
    fake = _FakeTreza({"make_image": [(401, {"error": "Invalid API key."})]})
    out = _tool(TrezaImageTool, fake).run(prompt="a red apple")

    assert "Invalid API key." in out
    assert "platform/api-keys" in out


def test_missing_key_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TREZA_API_KEY", raising=False)
    out = TrezaImageTool().run(prompt="a red apple")

    assert out.startswith("Error: Set TREZA_API_KEY or pass api_key.")
